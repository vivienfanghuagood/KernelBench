QUANT_OP_PROMPT = """
## 1. Hardware-Aware Optimization

### AMD MI300x/MI355 Constraints
- **Shared Memory (LDS)**: 64KB (65536 bytes) hardware limit
- **Matrix Cores**: MFMA instructions support FP8, FP16, BF16 natively
- **FP8 Format**: 
  - **gfx942 (MI300x)**: Use `torch.float8_e4m3fnuz` (AMD-specific)
  - **gfx950 (MI355+)**: Use `torch.float8_e4m3fn` or `torch.float8_e5m2` (OCP standard)
  - **Critical**: Using wrong FP8 format causes silent upcast to FP16, degrading performance significantly

### Memory Calculation Formula
```
Shared_Memory = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * dtype_size * num_stages
```

**Example:**
- Config: BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, num_stages=4, dtype=FP8 (1 byte)
- Memory: (256×64 + 256×64) × 1 × 4 = 131,072 bytes ❌ **EXCEEDS LIMIT**
- Solution: Reduce to BLOCK_M=128, BLOCK_N=128, num_stages=2 → 65,536 bytes ✓

### Native Precision Utilization

**Platform Detection for FP8:**
```python
import torch
import subprocess

def get_fp8_dtype():
    """Auto-detect correct FP8 dtype for current AMD GPU."""
    try:
        # Get GPU architecture
        result = subprocess.run(['rocminfo'], capture_output=True, text=True)
        output = result.stdout
        
        # Check for gfx950+ (MI355 and newer)
        if 'gfx950' in output or 'gfx960' in output:
            # Use OCP standard FP8 for newer architectures
            return torch.float8_e4m3fn
        else:
            # Use AMD-specific FP8 for gfx942 (MI300x) and older
            return getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)
    except:
        # Fallback: try AMD-specific first, then OCP standard
        return getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)

# Usage in kernel wrapper
fp8_dtype = get_fp8_dtype()
x_fp8 = x.to(fp8_dtype)
w_fp8 = w.to(fp8_dtype)
```

**DO:** Use native FP8 matrix operations with correct dtype
```python
x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0.0)  # Load as FP8
w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
result = tl.dot(x_fp8, tl.trans(w_fp8), out_dtype=tl.float32)  # FP8→FP32 matmul
```

**DON'T:** Convert to higher precision before computation
```python
x_fp16 = x_fp8.to(tl.float16)  # ❌ Wastes memory and compute
result = tl.dot(x_fp16, w_fp16)
```

**DON'T:** Use wrong FP8 format for target GPU
```python
# ❌ Will cause upcast to FP16 on gfx950 (MI355)
x_fp8 = x.to(torch.float8_e4m3fnuz)  # AMD-specific format
```

## 2. Autotuning Configuration Strategy

### Search Space Design
Target 6-10 configurations covering different workload characteristics:

```python
@triton.autotune(
    configs=[
        # Large tiles for throughput
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=8),
        
        # Smaller tiles for latency
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=4),
        
        # High parallelism
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=2, num_warps=16),
        
        # Pipeline optimization
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, 
                      num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
```

### Parameter Guidelines
- **BLOCK_M/N**: 64-256 (128-256 optimal for most cases)
- **BLOCK_K**: 64-128 (larger values amortize overhead in quantized ops)
- **num_stages**: 2-3 (AMD has stricter memory limits than NVIDIA)
- **num_warps**: 4-16 (8 is often optimal, scale with block size)
- **GROUP_M**: 8 (swizzling for L2 cache locality)

## 3. Kernel Structure Optimization

### Block Swizzling for Cache Locality
```python
# Compute with swizzling
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### Efficient Masking Pattern
```python
# Use separate masks for data and scaling
k_mask = (k + offs_k) < K
x_mask = (offs_m[:, None] < M) & k_mask[None, :]
w_mask = (offs_n[:, None] < N) & k_mask[None, :]

# Load with proper default values
x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0.0)  # Use 0.0, not 0
w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
```

**Critical:** Use `other=0.0` instead of `other=0` to avoid type casting errors with FP8.

### Accumulation Strategy
```python
# Always accumulate in FP32 for numerical stability
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

# Compute tiles and accumulate
for k in range(0, K, BLOCK_K):
    result = tl.dot(x_fp8, tl.trans(w_fp8), out_dtype=tl.float32)
    acc += result  # FP32 accumulation

# Convert to output dtype only at the end
out = acc.to(tl.float16)
```

## 4. Block-Wise Scaling Optimization

### Problem: Applying Per-Block Quantization Scales
Given:
- Input `x`: [M, K] in FP8
- Weight `w`: [N, K] in FP8
- Activation scales `x_scale`: [M, scale_k] where `scale_k = K // BLOCK_SIZE_K`
- Weight scales `w_scale`: [scale_n, scale_k] where `scale_n = N // BLOCK_SIZE_N`

### Naive Approach (SLOW ❌)
```python
# Load scales for entire K dimension: [BLOCK_M, BLOCK_K]
x_scale_vals = tl.load(x_scale_ptrs, ...)  # Repeated loads
w_scale_vals = tl.load(w_scale_ptrs, ...)

# Matrix multiplication for scaling (expensive!)
scale_matrix = tl.dot(x_scale_vals, tl.trans(w_scale_vals))  # [BLOCK_M, BLOCK_N]
acc += fp8_result * scale_matrix
```

### Optimized Approach (FAST ✓)
**Key Insight:** Scales are constant within each block, so we can use outer product:

```python
# Load scale blocks
scale_k_idx = (k + offs_k) // SCALE_BLOCK_K
x_scale_ptrs = x_scale_ptr + (offs_m[:, None] * stride_xscale_m + 
                               scale_k_idx[None, :] * stride_xscale_k)
x_scale_vals = tl.load(x_scale_ptrs, mask=x_scale_mask, other=1.0)

scale_n_idx = offs_n // SCALE_BLOCK_N
w_scale_ptrs = w_scale_ptr + (scale_n_idx[:, None] * stride_wscale_n + 
                               scale_k_idx[None, :] * stride_wscale_k)
w_scale_vals = tl.load(w_scale_ptrs, mask=w_scale_mask, other=1.0)

# Extract constant scale values (mean over repeated values)
x_scale_avg = tl.sum(x_scale_vals, axis=1) / BLOCK_K  # [BLOCK_M]
w_scale_avg = tl.sum(w_scale_vals, axis=1) / BLOCK_K  # [BLOCK_N]

# Outer product: [BLOCK_M, 1] × [1, BLOCK_N] = [BLOCK_M, BLOCK_N]
scale_factor = x_scale_avg[:, None] * w_scale_avg[None, :]

# Apply scaling (single element-wise multiplication)
acc += fp8_result * scale_factor
```

**Performance Impact:** 2-3x faster than matrix multiplication approach!

## 5. Common Pitfalls and Solutions

### Issue 1: Type Casting Errors
**Error:** `cannot cast int32 to fp8e4m3fnuz`
```python
x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0)  # ❌ other=0 is int32
```
**Solution:**
```python
x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0.0)  # ✓ other=0.0 is float
```

### Issue 2: Shared Memory Overflow
**Error:** `Allocation requires 122880 bytes but only 65536 available`

**Solution:** Reduce memory footprint
```python
# Before (exceeds limit)
BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, num_stages=4

# After (within limit)
BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_stages=2
```

### Issue 3: Precision Loss
**Problem:** Accumulated errors in long reduction chains

**Solution:** Use FP32 accumulation
```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # Not FP16!
result = tl.dot(x_fp8, w_fp8, out_dtype=tl.float32)   # Output in FP32
acc += result
out = acc.to(output_dtype)  # Convert only at the end
```

### Issue 4: Poor Performance with Small BLOCK_K
**Problem:** In quantized kernels, small BLOCK_K increases scaling overhead

**Solution:** Use larger BLOCK_K (64-128) to amortize the cost
```python
# Quantized ops: prefer BLOCK_K >= 64
triton.Config({'BLOCK_K': 128, ...})  # Better amortization
triton.Config({'BLOCK_K': 64, ...})   # Acceptable
triton.Config({'BLOCK_K': 32, ...})   # ❌ Too much overhead
```

### Issue 5: FP8 Format Mismatch (AMD GPUs)
**Error/Warning:** `fp8e4b8 is AMD gfx942 specific and not supported on gfx950 so it's upcasted to fp16`

**Problem:** Using `torch.float8_e4m3fnuz` (AMD-specific) on newer GPUs (MI355/gfx950+)

**Solution:** Detect GPU architecture and use correct FP8 format
```python
import subprocess

def get_fp8_dtype():
    """Auto-detect correct FP8 dtype for current AMD GPU."""
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True)
        output = result.stdout
        
        # gfx950+ uses OCP standard FP8
        if 'gfx950' in output or 'gfx960' in output:
            return torch.float8_e4m3fn  # OCP standard
        else:
            # gfx942 (MI300x) uses AMD-specific
            return getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)
    except:
        return getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)

# In get_inputs() function
fp8_dtype = get_fp8_dtype()
x = (torch.rand((m, k), dtype=torch.float16, device="cuda") / 10).to(fp8_dtype)
weight = (torch.rand((n, k), dtype=torch.float16, device="cuda") / 10).to(fp8_dtype)
```

**Alternative:** Environment-based configuration
```python
import os

def get_fp8_dtype():
    """Get FP8 dtype with optional override via environment variable."""
    # Allow manual override: export AITER_GPU_ARCH=gfx950
    gpu_arch = os.environ.get('AITER_GPU_ARCH', '')
    
    if 'gfx950' in gpu_arch or 'gfx960' in gpu_arch:
        return torch.float8_e4m3fn  # OCP standard for MI355+
    elif 'gfx942' in gpu_arch:
        return torch.float8_e4m3fnuz  # AMD-specific for MI300x
    else:
        # Auto-detect
        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True)
            if 'gfx950' in result.stdout or 'gfx960' in result.stdout:
                return torch.float8_e4m3fn
        except:
            pass
        return getattr(torch, 'float8_e4m3fnuz', torch.float8_e4m3fn)
```

## Summary

This guide provides a systematic approach to developing high-performance Triton kernels for AMD GPUs:

1. **Understand hardware constraints** (memory limits, native dtypes, GPU architecture differences)
2. **Use correct FP8 format** (gfx942: `float8_e4m3fnuz`, gfx950+: `float8_e4m3fn`)
3. **Design comprehensive autotuning** (6-10 configs, conservative num_stages)
4. **Structure kernels efficiently** (swizzling, masking, FP32 accumulation)
5. **Optimize fusion patterns** (outer products over matrix multiplications)
6. **Avoid common pitfalls** (type casting, memory overflow, precision loss, FP8 format mismatch)
7. **Validate rigorously** (correctness, performance, memory usage, check for upcast warnings)
"""

HIGH_CORRECT_PROMPT = """
### **CRITICAL: CORRECTNESS FIRST, PERFORMANCE SECOND** ###
You MUST guarantee the correctness of ModelNew. Do NOT cheat by simplifying logic or skipping computations.

### **Common Errors to AVOID** ###

**ERROR 1: Numerical Accuracy Issues (GELU, Normalization, Softmax)**
- ❌ WRONG: Using fp16 directly for `tl.exp()`, `tl.log()`, or complex math operations
- ✅ CORRECT: Always cast to fp32 for intermediate calculations, then cast back:
  ```python
  # For GELU, Softmax, LayerNorm, RMSNorm, etc.
  x_fp32 = x.to(tl.float32)
  result = compute_with_exp_log(x_fp32)  # Do math in fp32
  output = result.to(input_dtype)  # Cast back to original dtype
  ```
- For normalization operations (LayerNorm, RMSNorm, BatchNorm):
  * Compute mean and variance in fp32
  * Use numerically stable formulas: `variance = E[x²] - E[x]²`
  * Add epsilon BEFORE sqrt: `rsqrt(variance + eps)`

**ERROR 2: Missing @triton.jit Decorator**
- ❌ WRONG: Helper functions without decorator that use Triton operations
  ```python
  def helper(x):  # Missing decorator!
      return tl.exp(x)
  ```
- ✅ CORRECT: Add `@triton.jit` to ALL functions using Triton ops:
  ```python
  @triton.jit
  def helper(x):
      return tl.exp(x)
  ```

**ERROR 3: Invalid Triton APIs**
- ❌ FORBIDDEN APIs (do NOT use):
  * `tl.math.tanh` → Use: `(tl.exp(2*x) - 1) / (tl.exp(2*x) + 1)`
  * `tl.tanh` → Same as above
  * `tl.astype()` → Use: `.to(dtype)`
  * `tl.floor_div`, `tl.floor_divide` → Use: `tl.math.floor(x / y)` or `x // y`
  * `tl.full_like(x, v)` → Use: `tl.zeros_like(x) + v`
  * `tl.sum(x, where=...)` → NOT supported, use masking: `tl.sum(tl.where(mask, x, 0))`
  * `tl.program_id(axis=3)` → Only axes 0,1,2 supported (3D grid max)

**ERROR 4: Tensor Indexing Issues**
- ❌ WRONG: Using scalar indices on multi-dimensional tensors
  ```python
  qk += Q_block[:, k][:, None] * K_block[None, :, k]  # k is int32 scalar!
  ```
- ✅ CORRECT: Use proper slicing or tl.arange for indexing:
  ```python
  # Option 1: Expand dimensions properly
  q_vec = tl.load(Q_ptr + row_idx * stride + k)  # Load as 1D
  k_vec = tl.load(K_ptr + k * stride + col_idx)
  qk += q_vec[:, None] * k_vec[None, :]
  
  # Option 2: Use range-based indexing
  k_range = tl.arange(0, BLOCK_K) + k
  Q_slice = tl.load(Q_ptr + row_idx[:, None] * stride + k_range[None, :])
  ```

**ERROR 5: Control Flow Restrictions**
- ❌ FORBIDDEN: `continue` and `break` statements in Triton kernels
  ```python
  for i in range(N):
      if condition:
          continue  # NOT ALLOWED!
  ```
- ✅ CORRECT: Use conditional execution with tl.where or restructure logic:
  ```python
  for i in range(N):
      mask = condition
      result = tl.where(mask, compute_A(x), compute_B(x))
  ```

### **Mandatory Rules for Correctness** ###

1. **Dtype Handling:**
   - Cast to fp32 for: exp, log, sqrt, rsqrt, division, complex math
   - Keep original dtype for: loads, stores, simple add/multiply
   - Pattern: `input.to(tl.float32)` → compute → `.to(original_dtype)`

2. **Triton Decorator Requirements:**
   - ALWAYS add `@triton.jit` before any function using Triton language ops
   - Include `@triton.autotune` for performance optimization (optional but recommended)
   - All helper functions called from kernel need `@triton.jit`

3. **Memory and Indexing:**
   - `tl.arange()` arguments must be compile-time constants
   - Use contiguous pointers for `tl.store`, not block tensors directly
   - Apply bounds checking: `mask = (idx < N)` before load/store
   - For multi-dimensional indexing, use explicit offset calculations

4. **Numerical Stability:**
   - GELU: Use `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` in fp32
   - Softmax: Subtract max before exp: `exp(x - max(x))`
   - LayerNorm/RMSNorm: Compute statistics in fp32, add epsilon before rsqrt
   - Use `tl.math.fast_dividef` only when accuracy loss is acceptable

5. **Control Flow:**
   - Replace `continue` with conditional masks
   - Replace `break` with early termination flags and tl.where
   - Avoid Python control flow depending on Triton runtime values

6. **Hardware Constraints (AMD MI300x):**
   - Shared memory per block: ≤ 65536 bytes
   - Configure BLOCK_SIZE, num_stages, num_warps carefully
   - Use `@triton.autotune` to find optimal configs

7. **Weight Preprocessing:**
   - In `__init__`: transpose, reshape, convert weights to optimal layout
   - In `forward`: avoid dynamic transpose/reshape operations
   - Store preprocessed weights as nn.Parameter or buffers

### **Verification Checklist** ###
Before submitting ModelNew, verify:
- [ ] All Triton functions have `@triton.jit` decorator
- [ ] No use of forbidden APIs (tl.math.tanh, tl.astype, etc.)
- [ ] Floating-point ops use fp32 intermediate precision
- [ ] Tensor indexing uses proper dimensions (no scalar index on 2D tensor)
- [ ] No `break` or `continue` statements
- [ ] Bounds checking with masks before load/store
- [ ] Numerical computations match reference implementation exactly
- [ ] Memory usage within hardware limits

### **Priority: CORRECTNESS > PERFORMANCE** ###
If unsure, choose the more numerically stable and correct approach over performance optimization.
"""

HIGH_PERF_PROMPT = """
### **CRITICAL: EXTREME PERFORMANCE OPTIMIZATION** ###
You MUST achieve >2x speedup over PyTorch baseline while maintaining correctness. Use aggressive optimizations and fast approximations.

### **PERFORMANCE FIRST: Key Optimization Strategies** ###

**STRATEGY 1: Maximize Memory Bandwidth**
- ✅ ALWAYS use contiguous memory access patterns (coalesced loads/stores)
- ✅ Use vectorized loads: `tl.load()` with large BLOCK_SIZE (512, 1024, 2048)
- ✅ Minimize pointer arithmetic overhead
- ✅ For split operations (e.g., x[:, :d] and x[:, d:]), consider processing full rows at once
- ❌ AVOID strided access patterns like `x[::2]` or non-contiguous slicing

**STRATEGY 2: Kernel Fusion**
- ✅ Fuse ALL elementwise operations into a single kernel (e.g., GELU + multiply + add)
- ✅ Avoid intermediate memory allocations
- ✅ Perform all computations in-register when possible
- Example: `gelu(x) * y` should be ONE kernel, not two separate operations

**STRATEGY 3: Fast Math Approximations**
- ✅ GELU: Use `x * 0.5 * (1.0 + tl.math.tanh(0.797885 * (x + 0.044715 * x * x * x)))` OR faster sigmoid version
- ✅ For GELU, the FASTEST approximation: `x * tl.sigmoid(1.702 * x)` (3-5x faster)
- ✅ Tanh: Use `x * (27 + x²) / (27 + 9x²)` Padé approximation (no exp!)
- ✅ Sigmoid: Clamp input then use `1.0 / (1.0 + tl.exp(-x))`
- ✅ Use `tl.libdevice.fast_*` functions when available

**STRATEGY 4: Aggressive Block Configurations**
- ✅ Test BLOCK_SIZE from 256 to 4096
- ✅ Use num_warps=8 or 16 for large blocks
- ✅ Use num_stages=3 or 4 for memory-bound kernels (enables software pipelining)
- ✅ Always use `@triton.autotune` with at least 8-12 configurations

### **Common Errors to AVOID** ###

**ERROR 1: Numerical Accuracy Issues (GELU, Normalization, Softmax)**
- ❌ WRONG: Using fp16 directly for `tl.exp()`, `tl.log()`, or complex math operations
- ✅ CORRECT: Always cast to fp32 for intermediate calculations, then cast back:
  ```python
  # For GELU, Softmax, LayerNorm, RMSNorm, etc.
  x_fp32 = x.to(tl.float32)
  result = compute_with_exp_log(x_fp32)  # Do math in fp32
  output = result.to(input_dtype)  # Cast back to original dtype
  ```
- For normalization operations (LayerNorm, RMSNorm, BatchNorm):
  * Compute mean and variance in fp32
  * Use numerically stable formulas: `variance = E[x²] - E[x]²`
  * Add epsilon BEFORE sqrt: `rsqrt(variance + eps)`

**ERROR 2: Missing @triton.jit Decorator**
- ❌ WRONG: Helper functions without decorator that use Triton operations
  ```python
  def helper(x):  # Missing decorator!
      return tl.exp(x)
  ```
- ✅ CORRECT: Add `@triton.jit` to ALL functions using Triton ops:
  ```python
  @triton.jit
  def helper(x):
      return tl.exp(x)
  ```

**ERROR 3: Invalid Triton APIs**
- ❌ FORBIDDEN APIs (do NOT use):
  * `tl.math.tanh` → Use fast approximations (see below)
  * `tl.tanh` → Use fast approximations (see below)
  * `tl.astype()` → Use: `.to(dtype)`
  * `tl.floor_div`, `tl.floor_divide` → Use: `x // y`
  * `tl.full_like(x, v)` → Use: `tl.zeros_like(x) + v`
  * `tl.sum(x, where=...)` → NOT supported, use masking: `tl.sum(tl.where(mask, x, 0))`
  * `tl.program_id(axis=3)` → Only axes 0,1,2 supported (3D grid max)

**ERROR 4: Inefficient Memory Access Patterns**
- ❌ WRONG: Non-coalesced loads (strided, scattered)
  ```python
  # BAD: Each thread loads from non-contiguous locations
  offsets = row_idx * stride + col_idx  # If threads have different row_idx
  data = tl.load(ptr + offsets)
  ```
- ✅ CORRECT: Coalesced loads (contiguous blocks)
  ```python
  # GOOD: All threads in a warp load contiguous memory
  block_start = tl.program_id(0) * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Contiguous!
  data = tl.load(ptr + offsets, mask=offsets < N)
  ```

**ERROR 5: Tensor Indexing Issues**
- ❌ WRONG: Using scalar indices on multi-dimensional tensors
  ```python
  qk += Q_block[:, k][:, None] * K_block[None, :, k]  # k is int32 scalar!
  ```
- ✅ CORRECT: Use proper slicing or tl.arange for indexing:
  ```python
  # Option 1: Expand dimensions properly
  q_vec = tl.load(Q_ptr + row_idx * stride + k)  # Load as 1D
  k_vec = tl.load(K_ptr + k * stride + col_idx)
  qk += q_vec[:, None] * k_vec[None, :]
  
  # Option 2: Use range-based indexing
  k_range = tl.arange(0, BLOCK_K) + k
  Q_slice = tl.load(Q_ptr + row_idx[:, None] * stride + k_range[None, :])
  ```

**ERROR 6: Control Flow Restrictions**
- ❌ FORBIDDEN: `continue` and `break` statements in Triton kernels
  ```python
  for i in range(N):
      if condition:
          continue  # NOT ALLOWED!
  ```
- ✅ CORRECT: Use conditional execution with tl.where or restructure logic:
  ```python
  for i in range(N):
      mask = condition
      result = tl.where(mask, compute_A(x), compute_B(x))
  ```

### **ULTRA-FAST Activation Functions** ###

**1. GELU - Multiple Speed Tiers (CRITICAL for >2x speedup):**
```python
# FASTEST: Sigmoid approximation - ALWAYS use this by default!
@triton.jit
def ultra_fast_gelu(x):
    # gelu(x) ≈ x * σ(1.702 * x)
    # Only ONE exp operation, 3-5x faster than tanh version
    return x * tl.sigmoid(1.702 * x)

# ALTERNATIVE: Tanh with Padé approximation (NO exp!)
@triton.jit  
def fast_gelu_no_exp(x):
    # gelu(x) ≈ 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x³)))
    x3 = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x3)
    # Padé: tanh(x) ≈ x(27 + x²) / (27 + 9x²)
    x2 = inner * inner
    tanh_val = inner * (27.0 + x2) / (27.0 + 9.0 * x2)
    return 0.5 * x * (1.0 + tanh_val)
```

**2. Tanh - Ultra Fast (NO exp needed):**
```python
@triton.jit
def ultra_fast_tanh(x):
    # Padé [3/3] approximation - NO exp operations!
    # tanh(x) ≈ x(27 + x²) / (27 + 9x²)
    x_clamped = tl.where(x > 5.0, 5.0, tl.where(x < -5.0, -5.0, x))
    x2 = x_clamped * x_clamped
    return x_clamped * (27.0 + x2) / (27.0 + 9.0 * x2)
```

### **CRITICAL Performance Rules for >2x Speedup** ###

1. **Memory Access Optimization (HIGHEST PRIORITY):**
   - ✅ ALWAYS process contiguous blocks: `offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`
   - ✅ For split tensors like `x[:, :d]` and `x[:, d:]`, flatten to 1D:
     ```python
     # BEST PRACTICE for gelu(x[:, :d]) * x[:, d:]:
     N = batch_size * out_features
     flat_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     row = flat_idx // out_features
     col = flat_idx % out_features
     gate_idx = row * (2 * out_features) + col  # Fully coalesced!
     up_idx = row * (2 * out_features) + out_features + col
     gate = tl.load(x_ptr + gate_idx, mask=flat_idx < N)
     up = tl.load(x_ptr + up_idx, mask=flat_idx < N)
     ```
   - ✅ Use LARGE BLOCK_SIZE: 1024, 2048, 4096 (test aggressively!)
   - ❌ AVOID strided or scattered access patterns

2. **Aggressive Autotuning (MANDATORY):**
   ```python
   @triton.autotune(
       configs=[
           triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=3),
           triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
           triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
           triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
           triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
           triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
           # Try different combinations
           triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
           triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
       ],
       key=['N'],
   )
   ```

3. **Kernel Fusion (ESSENTIAL):**
   - Fuse ALL elementwise ops into ONE kernel
   - Example: `gelu(x) * y` should load both, compute, store once
   - Avoid intermediate memory writes

4. **Fast Math:**
   - GELU: ALWAYS use `x * tl.sigmoid(1.702 * x)` by default
   - Tanh: Use Padé approximation (no exp)
   - Use fp32 for math, but minimize conversions

5. **Triton Decorator Requirements:**
   - ALWAYS add `@triton.jit` before any function using Triton language ops
   - ALWAYS use `@triton.autotune` for performance-critical kernels
   - All helper functions called from kernel need `@triton.jit`

6. **Memory and Indexing Optimization:**
   - `tl.arange()` arguments must be compile-time constants
   - Use contiguous pointers for `tl.store`, not block tensors directly
   - Apply bounds checking: `mask = (idx < N)` before load/store
   - For multi-dimensional indexing, use explicit offset calculations
   - Maximize memory coalescing: load contiguous blocks when possible
   - Use larger BLOCK_SIZE for better occupancy (e.g., 1024, 2048, 4096)

7. **Control Flow:**
   - Replace `continue` with conditional masks
   - Replace `break` with early termination flags and tl.where
   - Avoid Python control flow depending on Triton runtime values

8. **Hardware Optimization:**
   - Shared memory per block: ≤ 65536 bytes
   - Configure BLOCK_SIZE, num_stages, num_warps for maximum throughput
   - num_warps: 4-16 (higher for larger BLOCK_SIZE)
   - num_stages: 2-4 for software pipelining (more overlap = better performance)
   - Test configurations aggressively with `@triton.autotune`

9. **Weight Preprocessing:**
   - In `__init__`: transpose, reshape, convert weights to optimal layout
   - In `forward`: avoid dynamic transpose/reshape operations
   - Store preprocessed weights as nn.Parameter or buffers

### **Performance Checklist for >2x Speedup** ###
Before submitting ModelNew, verify:
- [ ] All Triton functions have `@triton.jit` decorator
- [ ] No use of forbidden APIs (tl.math.tanh, tl.astype, etc.)
- [ ] Using ultra_fast_gelu (sigmoid version) by default
- [ ] Memory access is FULLY COALESCED (contiguous blocks)
- [ ] BLOCK_SIZE includes 1024, 2048, 4096 in autotune
- [ ] At least 6-8 autotune configurations tested
- [ ] num_stages >= 3 for memory-bound kernels
- [ ] Tensor indexing uses proper dimensions
- [ ] No `break` or `continue` statements
- [ ] Operations are FUSED (one kernel for multi-step ops)
- [ ] Minimal dtype conversions (fp32 only for math ops)

### **Priority: >2x SPEEDUP TARGET** ###
Default to the FASTEST implementations (sigmoid GELU, Padé tanh, large BLOCK_SIZE, aggressive autotuning).
Focus on memory bandwidth optimization and kernel fusion.
"""