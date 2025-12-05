# AMD RDNA4 Triton Kernel Generation Guide

## Overview

This guide is specifically designed for generating correct and high-performance Triton kernels on AMD RDNA4 GPUs (e.g., RX 9000 series). RDNA4 architecture differs significantly from CDNA (MI300x/MI355) in wavefront size, compute unit design, and optimal parameter configurations.

---

## 1. RDNA4 Architecture Fundamentals

### Key Differences from CDNA

| Feature | RDNA4 | CDNA (MI300x) |
|---------|-------|---------------|
| Wavefront Size | **Wave32 (32 threads)** | Wave64 (64 threads) |
| Matrix Cores | No native FP8 MFMA | Full MFMA support |
| Primary Focus | Gaming/Consumer | HPC/AI |
| LDS Per Workgroup | 64KB | 64KB |
| Optimal num_warps | 1-8 (wave32-based) | 4-16 (wave64-based) |

### Critical: Wave32 Architecture

RDNA4 uses **Wave32** (32 work-items per wavefront), which fundamentally affects `num_warps` configuration:

```python
# RDNA4: Each "warp" in Triton actually maps to a Wave32 (32 threads)
# For a BLOCK_SIZE of 256:
#   - RDNA4: needs 256/32 = 8 wavefronts → num_warps should be 4-8
#   - CDNA:  needs 256/64 = 4 wavefronts → num_warps would be 4
```

**Rule of Thumb for num_warps on RDNA4:**
- BLOCK_SIZE 64-128: `num_warps=2-4`
- BLOCK_SIZE 256: `num_warps=4-8`
- BLOCK_SIZE 512-1024: `num_warps=8`

---

## 2. Mandatory Correctness Rules (CRITICAL)

### 2.1 Numerical Precision - ALWAYS Use FP32 for Math Operations

**This is the #1 cause of correctness failures on RDNA4.**

```python
# ❌ WRONG: Direct FP16 math operations cause precision loss
@triton.jit
def bad_gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask)
    # Direct operations in fp16 - WILL FAIL CORRECTNESS
    result = x * 0.5 * (1.0 + tl.libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(out_ptr + idx, result, mask=mask)

# ✅ CORRECT: Cast to FP32, compute, cast back
@triton.jit
def correct_gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask)
    
    # Cast to FP32 for ALL math operations
    x_fp32 = x.to(tl.float32)
    
    # Perform math in FP32
    inner = 0.7978845608 * (x_fp32 + 0.044715 * x_fp32 * x_fp32 * x_fp32)
    tanh_val = tl.libdevice.tanh(inner)
    result_fp32 = x_fp32 * 0.5 * (1.0 + tanh_val)
    
    # Cast back to original dtype
    result = result_fp32.to(x.dtype)
    tl.store(out_ptr + idx, result, mask=mask)
```

### 2.2 Required @triton.jit Decorator

**ALL functions using Triton operations MUST have @triton.jit:**

```python
# ❌ WRONG: Missing decorator
def compute_gelu(x):
    return x * 0.5 * (1.0 + tl.libdevice.tanh(...))

# ✅ CORRECT: With decorator
@triton.jit
def compute_gelu(x):
    return x * 0.5 * (1.0 + tl.libdevice.tanh(...))
```

### 2.3 Forbidden Triton APIs (DO NOT USE)

These APIs either don't exist or behave incorrectly on AMD GPUs:

| Forbidden API | Correct Alternative |
|---------------|---------------------|
| `tl.math.tanh` | `tl.libdevice.tanh` (with FP32 input) |
| `tl.tanh` | `tl.libdevice.tanh` (with FP32 input) |
| `tl.astype()` | `.to(dtype)` |
| `tl.floor_div` | `x // y` or `tl.math.floor(x / y).to(tl.int32)` |
| `tl.full_like(x, v)` | `tl.zeros_like(x) + v` |
| `tl.sum(x, where=...)` | `tl.sum(tl.where(mask, x, 0.0))` |
| `tl.program_id(axis=3)` | Only 0, 1, 2 axes supported |

### 2.4 Correct Use of tl.load/tl.store

```python
# ❌ WRONG: Integer default value for float data
x = tl.load(ptr + idx, mask=mask, other=0)  # other=0 is int32!

# ✅ CORRECT: Float default value
x = tl.load(ptr + idx, mask=mask, other=0.0)  # other=0.0 is float
```

### 2.5 Control Flow Restrictions

**NEVER use `break` or `continue` in Triton kernels:**

```python
# ❌ WRONG: Using break/continue
for i in range(N):
    if condition:
        continue  # NOT ALLOWED

# ✅ CORRECT: Use tl.where instead
for i in range(N):
    result = tl.where(condition, skip_value, compute_value)
```

---

## 3. RDNA4-Optimized Autotuning Configurations

### 3.1 Elementwise Operations (GELU, ReLU, Sigmoid, etc.)

```python
@triton.autotune(
    configs=[
        # Conservative configs for correctness
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        # Performance-focused configs
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def elementwise_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

### 3.2 Reduction Operations (Sum, Mean, Max, Softmax, Norm)

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def reduction_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...
```

### 3.3 Matrix Operations (MatMul, GEMM)

**Note: RDNA4 does NOT have MFMA (Matrix Fused Multiply-Add) instructions like CDNA. Use standard tl.dot with smaller tiles.**

```python
@triton.autotune(
    configs=[
        # Smaller tiles for RDNA4 (no matrix cores)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

### 3.4 Parameter Guidelines for RDNA4

| Parameter | Recommended Values | Notes |
|-----------|-------------------|-------|
| BLOCK_SIZE (1D) | 256, 512, 1024 | Start with 512 |
| BLOCK_M/N (2D) | 64, 128 | Avoid 256+ without matrix cores |
| BLOCK_K | 32, 64 | Smaller than CDNA |
| num_warps | 2-8 | Due to Wave32 |
| num_stages | 2-3 | Conservative for AMD |
| GROUP_M | 8 | For L2 cache locality |

---

## 4. Correct Implementation Patterns

### 4.1 GELU Activation (Level 1: 26_GELU_.py)

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # CRITICAL: Cast to FP32 for numerical accuracy
    x_fp32 = x.to(tl.float32)
    
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # sqrt(2/pi) ≈ 0.7978845608
    x3 = x_fp32 * x_fp32 * x_fp32
    inner = 0.7978845608 * (x_fp32 + 0.044715 * x3)
    
    # Use tl.libdevice.tanh for AMD compatibility
    tanh_val = tl.libdevice.tanh(inner)
    result_fp32 = x_fp32 * 0.5 * (1.0 + tanh_val)
    
    # Cast back to original dtype
    result = result_fp32.to(x.dtype)
    
    tl.store(out_ptr + offsets, result, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(x)
        N = x.numel()
        
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        gelu_kernel[grid](x, output, N)
        
        return output
```

### 4.2 Softmax (Level 1: 23_Softmax.py)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load row data in chunks, find max for numerical stability
    max_val = float('-inf')
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < n_cols
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=float('-inf'))
        # Cast to FP32 for precision
        x_fp32 = x.to(tl.float32)
        max_val = tl.maximum(max_val, tl.max(x_fp32, axis=0))
    
    # Compute exp(x - max) and sum
    sum_exp = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < n_cols
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=float('-inf'))
        x_fp32 = x.to(tl.float32)
        exp_x = tl.exp(x_fp32 - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_x, 0.0), axis=0)
    
    # Compute softmax and store
    for start in range(0, n_cols, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < n_cols
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=float('-inf'))
        x_fp32 = x.to(tl.float32)
        exp_x = tl.exp(x_fp32 - max_val)
        softmax_val = exp_x / sum_exp
        # Cast back to original dtype
        tl.store(out_ptr + row_start + offs, softmax_val.to(x.dtype), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_cols = x.shape
        output = torch.empty_like(x)
        
        # Choose BLOCK_SIZE based on n_cols
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4096)
        
        grid = (batch_size,)
        softmax_kernel[grid](
            x, output, n_cols, x.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4 if BLOCK_SIZE <= 512 else 8,
            num_stages=2,
        )
        
        return output
```

### 4.3 LayerNorm (Level 1: 40_LayerNorm.py)

```python
import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    M,  # Number of rows
    N,  # Number of features (normalized dimension)
    eps,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * stride
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute mean (in FP32)
    mean = 0.0
    for start in range(0, N, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < N
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        mean += tl.sum(tl.where(mask, x_fp32, 0.0), axis=0)
    mean = mean / N
    
    # Compute variance (in FP32)
    var = 0.0
    for start in range(0, N, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < N
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        diff = tl.where(mask, x_fp32 - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    
    # Compute normalized output
    rstd = tl.rsqrt(var + eps)
    
    for start in range(0, N, BLOCK_SIZE):
        offs = start + col_offsets
        mask = offs < N
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        
        # Load gamma and beta
        gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        
        # Normalize
        normalized = (x_fp32 - mean) * rstd
        out = normalized * gamma + beta
        
        # Cast back to original dtype
        tl.store(out_ptr + row_start + offs, out.to(x.dtype), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = 1e-5
        # Create learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten last dimensions for normalization
        orig_shape = x.shape
        N = 1
        for dim in self.normalized_shape:
            N *= dim
        M = x.numel() // N
        
        x_flat = x.view(M, N)
        output = torch.empty_like(x_flat)
        
        BLOCK_SIZE = min(triton.next_power_of_2(N), 4096)
        
        grid = (M,)
        layernorm_kernel[grid](
            x_flat, output,
            self.gamma.flatten(), self.beta.flatten(),
            M, N, self.eps, N,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4 if BLOCK_SIZE <= 512 else 8,
            num_stages=2,
        )
        
        return output.view(orig_shape)
```

### 4.4 Matrix Multiplication (Level 1: 2_Standard_matrix_multiplication_.py)

```python
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # CRITICAL: Accumulate in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Cast to FP32 for accumulation
        acc += tl.dot(a.to(tl.float32), b.to(tl.float32))
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Store with original dtype
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )
        
        matmul_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
        )
        
        return C
```

---

## 5. Level 2 Fused Operations

### 5.1 Fused Matmul + GELU + Softmax (Level 2: 99_Matmul_GELU_Softmax.py)

For Level 2 operations, consider fusion strategies but prioritize correctness:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_gelu_kernel(
    x_ptr, out_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)
    
    # GELU
    x3 = x_fp32 * x_fp32 * x_fp32
    inner = 0.7978845608 * (x_fp32 + 0.044715 * x3)
    tanh_val = tl.libdevice.tanh(inner)
    gelu_out = x_fp32 * 0.5 * (1.0 + tanh_val)
    
    tl.store(out_ptr + offs, gelu_out.to(x.dtype), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        # Use PyTorch for matmul (well-optimized)
        x = self.linear(x)
        
        # Custom Triton GELU
        output = torch.empty_like(x)
        N = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        fused_gelu_kernel[grid](x, output, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        
        # Use PyTorch for softmax (numerically stable)
        output = torch.nn.functional.softmax(output, dim=1)
        
        return output
```

**Strategy for Level 2:**
1. Identify which operations benefit from custom kernels (elementwise, simple reductions)
2. Keep complex operations (matmul, softmax with large reduction) in PyTorch if correctness issues arise
3. Focus on fusing consecutive elementwise operations

---

## 6. Common Correctness Issues and Solutions

### Issue 1: NaN/Inf in Output

**Cause:** FP16 overflow/underflow in math operations

**Solution:**
```python
# Always cast to FP32 before exp, log, tanh, sqrt
x_fp32 = x.to(tl.float32)
result = tl.exp(x_fp32)  # Safe
result = result.to(x.dtype)  # Cast back
```

### Issue 2: Shape Mismatch

**Cause:** Incorrect stride calculations

**Solution:**
```python
# Always use explicit stride parameters
stride_row = x.stride(0)
stride_col = x.stride(1)

# Use strides in pointer arithmetic
ptr = base_ptr + row_idx * stride_row + col_idx * stride_col
```

### Issue 3: Incorrect Reduction Results

**Cause:** Wrong initialization or accumulation

**Solution:**
```python
# Initialize accumulators appropriately
sum_val = 0.0  # For sum
max_val = float('-inf')  # For max
min_val = float('inf')  # For min

# Use tl.where for masked operations
sum_val = tl.sum(tl.where(mask, x, 0.0), axis=0)
```

### Issue 4: Compilation Errors

**Cause:** Using unsupported APIs or wrong types

**Solution:**
- Check the forbidden APIs list above
- Use `.to(dtype)` instead of `.astype()`
- Use `tl.libdevice.tanh` instead of `tl.math.tanh`

---

## 7. Verification Checklist

Before submitting generated kernel, verify:

- [ ] All `@triton.jit` decorators present on kernel functions and helpers
- [ ] All math operations (exp, log, tanh, sqrt, rsqrt) use FP32 intermediate values
- [ ] No forbidden APIs used (tl.math.tanh, tl.astype, etc.)
- [ ] `tl.load` uses `other=0.0` (float) not `other=0` (int)
- [ ] No `break` or `continue` statements in kernel code
- [ ] `num_warps` values are appropriate for Wave32 (typically 2-8)
- [ ] `num_stages` is conservative (2-3 for AMD)
- [ ] Block sizes are reasonable (256-1024 for elementwise, 64-128 for tiles)
- [ ] Masks applied correctly for all loads/stores
- [ ] Output dtype matches input dtype after computation

---

## 8. Summary

For RDNA4, prioritize:

1. **Correctness over performance** - Always use FP32 for intermediate math
2. **Use `tl.libdevice.*` functions** - AMD-compatible math operations
3. **Conservative autotuning** - Smaller block sizes, fewer warps than CDNA
4. **Wave32 awareness** - `num_warps` should be 2-8, not 4-16
5. **Verify forbidden APIs** - Don't use `tl.math.tanh`, `tl.astype`, etc.
6. **Test thoroughly** - RDNA4 has different numerical characteristics than CDNA

