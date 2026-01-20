"""
Prompt templates for KernelBench AMD GPU guidance.

These prompts guide the LLM in generating optimized GPU kernels.
Supports both Triton and Helion DSLs.
"""

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
~~~
Shared_Memory = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * dtype_size * num_stages
~~~

**Example:**
- Config: BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, num_stages=4, dtype=FP8 (1 byte)
- Memory: (256×64 + 256×64) × 1 × 4 = 131,072 bytes ❌ **EXCEEDS LIMIT**
- Solution: Reduce to BLOCK_M=128, BLOCK_N=128, num_stages=2 → 65,536 bytes ✓

### Native Precision Utilization

**Platform Detection for FP8:**
~~~
import torch
import subprocess

def get_fp8_dtype():
    # Auto-detect correct FP8 dtype for current AMD GPU.
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
~~~

**DO:** Use native FP8 matrix operations with correct dtype
~~~
x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0.0)  # Load as FP8
w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
result = tl.dot(x_fp8, tl.trans(w_fp8), out_dtype=tl.float32)  # FP8→FP32 matmul
~~~

**DON'T:** Convert to higher precision before computation
~~~
x_fp16 = x_fp8.to(tl.float16)  # ❌ Wastes memory and compute
result = tl.dot(x_fp16, w_fp16)
~~~

## 2. Autotuning Configuration Strategy

### Search Space Design
Target 6-10 configurations covering different workload characteristics:

~~~
@triton.autotune(
    configs=[
        # Large tiles for throughput
        triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}},
                      num_stages=2, num_warps=8),
        triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}},
                      num_stages=2, num_warps=8),

        # Smaller tiles for latency
        triton.Config({{'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}},
                      num_stages=2, num_warps=4),
        triton.Config({{'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}},
                      num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
~~~

### Parameter Guidelines
- **BLOCK_M/N**: 64-256 (128-256 optimal for most cases)
- **BLOCK_K**: 64-128 (larger values amortize overhead in quantized ops)
- **num_stages**: 2-3 (AMD has stricter memory limits than NVIDIA)
- **num_warps**: 4-16 (8 is often optimal, scale with block size)
- **GROUP_M**: 8 (swizzling for L2 cache locality)

## 3. Common Pitfalls and Solutions

### Issue 1: Type Casting Errors
**Error:** `cannot cast int32 to fp8e4m3fnuz`
~~~
x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0)  # ❌ other=0 is int32
~~~
**Solution:**
~~~
x_fp8 = tl.load(x_ptrs, mask=x_mask, other=0.0)  # ✓ other=0.0 is float
~~~

### Issue 2: Shared Memory Overflow
**Error:** `Allocation requires 122880 bytes but only 65536 available`
**Solution:** Reduce memory footprint by using smaller block sizes or fewer stages.
"""

HIGH_CORRECT_PROMPT = """
### **CRITICAL: CORRECTNESS FIRST, PERFORMANCE SECOND** ###
You MUST guarantee the correctness of ModelNew. Do NOT cheat by simplifying logic or skipping computations.

### **Common Errors to AVOID** ###

**ERROR 1: Numerical Accuracy Issues (GELU, Normalization, Softmax)**
- ❌ WRONG: Using fp16 directly for `tl.exp()`, `tl.log()`, or complex math operations
- ✅ CORRECT: Always cast to fp32 for intermediate calculations, then cast back:
  ```
  # For GELU, Softmax, LayerNorm, RMSNorm, etc.
  x_fp32 = x.to(tl.float32)
  result = compute_with_exp_log(x_fp32)  # Do math in fp32
  output = result.to(input_dtype)  # Cast back to original dtype
  ```

**ERROR 2: Missing @triton.jit Decorator**
- ❌ WRONG: Helper functions without decorator that use Triton operations
- ✅ CORRECT: Add `@triton.jit` to ALL functions using Triton ops

**ERROR 3: Invalid Triton APIs**
- ❌ FORBIDDEN APIs (do NOT use):
  * `tl.math.tanh` → Use: `(tl.exp(2*x) - 1) / (tl.exp(2*x) + 1)`
  * `tl.astype()` → Use: `.to(dtype)`
  * `tl.floor_div`, `tl.floor_divide` → Use: `x // y`
  * `tl.full_like(x, v)` → Use: `tl.zeros_like(x) + v`

**ERROR 4: Tensor Indexing Issues**
- ❌ WRONG: Using scalar indices on multi-dimensional tensors
- ✅ CORRECT: Use proper slicing or tl.arange for indexing

**ERROR 5: Control Flow Restrictions**
- ❌ FORBIDDEN: `continue` and `break` statements in Triton kernels
- ✅ CORRECT: Use conditional execution with tl.where

### **Mandatory Rules for Correctness** ###

1. **Dtype Handling:**
   - Cast to fp32 for: exp, log, sqrt, rsqrt, division, complex math
   - Keep original dtype for: loads, stores, simple add/multiply

2. **Triton Decorator Requirements:**
   - ALWAYS add `@triton.jit` before any function using Triton language ops

3. **Numerical Stability:**
   - GELU: Use proper formula in fp32
   - Softmax: Subtract max before exp: `exp(x - max(x))`
   - LayerNorm/RMSNorm: Compute statistics in fp32, add epsilon before rsqrt

### **Priority: CORRECTNESS > PERFORMANCE** ###
If unsure, choose the more numerically stable and correct approach over performance optimization.
"""

HIGH_PERF_PROMPT = """
### **CRITICAL: EXTREME PERFORMANCE OPTIMIZATION** ###
You MUST achieve >2x speedup over PyTorch baseline while maintaining correctness.

### **PERFORMANCE FIRST: Key Optimization Strategies** ###

**STRATEGY 1: Maximize Memory Bandwidth**
- ✅ ALWAYS use contiguous memory access patterns (coalesced loads/stores)
- ✅ Use vectorized loads with large BLOCK_SIZE (512, 1024, 2048)
- ❌ AVOID strided access patterns

**STRATEGY 2: Kernel Fusion**
- ✅ Fuse ALL elementwise operations into a single kernel
- ✅ Avoid intermediate memory allocations

**STRATEGY 3: Fast Math Approximations**
- ✅ GELU FASTEST: `x * tl.sigmoid(1.702 * x)` (3-5x faster)
- ✅ Tanh: Use Padé approximation (no exp!)

**STRATEGY 4: Aggressive Block Configurations**
- ✅ Test BLOCK_SIZE from 256 to 4096
- ✅ Use num_warps=8 or 16 for large blocks
- ✅ Always use `@triton.autotune` with 8-12 configurations

### **ULTRA-FAST Activation Functions** ###

```python
# FASTEST GELU: Sigmoid approximation
@triton.jit
def ultra_fast_gelu(x):
    return x * tl.sigmoid(1.702 * x)

# FAST Tanh: Padé approximation (NO exp!)
@triton.jit
def ultra_fast_tanh(x):
    x2 = x * x
    return x * (27.0 + x2) / (27.0 + 9.0 * x2)
```

### **Priority: >2x SPEEDUP TARGET** ###
Default to the FASTEST implementations. Focus on memory bandwidth optimization and kernel fusion.
"""

HIGH_PERF_TRITON_PROMPT = """
mod1.15-2026-01-15
### **CRITICAL: TRITON→TRITON OPTIMIZATION ONLY** ###
You MUST optimize existing Triton kernels and wrappers. Do NOT switch to PyTorch ops, CUDA extensions, or other DSLs.
Keep the original API, tensor shapes, and semantics exactly identical. Performance is priority, but correctness is mandatory.

### **PRIMARY GOAL: TRITON→TRITON SPEEDUP** ###
- ✅ Rewrite/optimize Triton kernels and launch configs only
- ✅ Preserve I/O contracts (input dtypes, output dtypes, shapes)
- ✅ Keep numerically equivalent behavior unless explicitly allowed to approximate
- ❌ Do NOT add PyTorch fallback paths or replace kernels with torch ops
- ❌ Do NOT change model forward semantics, only implementation details

### **REQUIRED OUTPUT FORMAT** ###
1) Provide optimized Triton kernel code and wrapper code.
2) Use `@triton.autotune` with 6-12 configs.
3) Provide chosen block sizes, warps, stages, and explain why.
4) Keep the same class/function names and signatures as the original.

### **PERFORMANCE FIRST: Triton-Specific Strategies** ###

**STRATEGY 1: Memory Bandwidth and Coalescing**
- ✅ Prefer contiguous access with pointer arithmetic
- ✅ Use `tl.multiple_of` and `tl.assume` on strides when safe
- ✅ Use vectorized loads/stores (`tl.load` with block pointers)
- ❌ Avoid strided or scattered loads

**STRATEGY 2: Kernel Fusion**
- ✅ Fuse elementwise chains into one kernel
- ✅ Avoid intermediate tensors and extra kernel launches

**STRATEGY 3: Autotuning**
- ✅ Use `@triton.autotune` with 6-12 configs
- ✅ Test BLOCK sizes spanning 128-2048 depending on op
- ✅ Tune `num_warps` (4/8/16) and `num_stages` (2/3)

**STRATEGY 4: Math Precision**
- ✅ Use fp32 for exp/log/softmax/gelu internal math when required
- ✅ Cast back to original dtype for stores
- ✅ Use faster approximations only if acceptable for correctness

### **ULTRA-FAST Activation Functions (if allowed)** ###

```python
# FAST GELU: Sigmoid approximation
@triton.jit
def fast_gelu(x):
    return x * tl.sigmoid(1.702 * x)
```

### **Priority: TRITON→TRITON SPEEDUP** ###
Deliver the fastest Triton kernel(s) possible while preserving correctness and API.
"""

RDNA4_PROMPT = """
## AMD RDNA4 Triton Kernel Generation Guide

### CRITICAL: RDNA4 Architecture Differences from CDNA

**RDNA4 uses Wave32 (32 threads per wavefront) NOT Wave64 like CDNA (MI300x/MI355).**

| Feature | RDNA4 | CDNA (MI300x) |
|---------|-------|---------------|
| Wavefront Size | **Wave32 (32 threads)** | Wave64 (64 threads) |
| Matrix Cores | No native FP8 MFMA | Full MFMA support |
| Optimal num_warps | 2-8 | 4-16 |
| Optimal BLOCK_SIZE | 256-1024 | 128-2048 |

## 1. MANDATORY CORRECTNESS RULES (CRITICAL)

### 1.1 ALWAYS Use FP32 for Math Operations

**ALL math operations (exp, log, tanh, sqrt, division) MUST use FP32 intermediate values.**

~~~python
# ✅ CORRECT - Cast to FP32, compute, cast back
@triton.jit
def correct_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)

    # CRITICAL: Cast to FP32
    x_fp32 = x.to(tl.float32)
    result_fp32 = compute_in_fp32(x_fp32)
    result = result_fp32.to(x.dtype)

    tl.store(out_ptr + idx, result, mask=mask)
~~~

### 1.2 Forbidden APIs - DO NOT USE

| Forbidden API | Use Instead |
|---------------|-------------|
| `tl.math.tanh` | `tl.libdevice.tanh` (with FP32 input) |
| `tl.astype()` | `.to(dtype)` |
| `tl.floor_div` | `x // y` |
| `tl.full_like(x, v)` | `tl.zeros_like(x) + v` |

### 1.3 No break/continue Statements

Use `tl.where` for conditional execution instead.

## 2. RDNA4-Optimized Autotuning

~~~python
@triton.autotune(
    configs=[
        triton.Config({{'BLOCK_SIZE': 256}}, num_warps=4, num_stages=2),
        triton.Config({{'BLOCK_SIZE': 512}}, num_warps=4, num_stages=2),
        triton.Config({{'BLOCK_SIZE': 512}}, num_warps=8, num_stages=2),
        triton.Config({{'BLOCK_SIZE': 1024}}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
~~~

### Parameter Guidelines for RDNA4

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| BLOCK_SIZE (1D) | 256-1024 | Start with 512 |
| BLOCK_M/N (2D) | 64-128 | No 256+ without matrix cores |
| num_warps | 2-8 | Due to Wave32 |
| num_stages | 2-3 | Conservative |

### Priority: CORRECTNESS > PERFORMANCE

On RDNA4, always prioritize generating correct code. Use FP32 intermediate precision everywhere.
"""

HELION_PROMPT = """
## Helion Kernel Generation Guide

Helion is a high-level DSL for writing GPU kernels that compiles to Triton. It provides a more Pythonic
interface with automatic tiling and memory management.

### Key Helion Concepts

1. **Kernel Decorator**: Use `@helion.kernel()` to define a Helion kernel
2. **Tiling**: Use `hl.tile([m, n])` to create tile iterators over dimensions
3. **Accumulation**: Use `hl.zeros([tile_m, tile_n], dtype=torch.float32)` for accumulators
4. **Matrix Operations**: Use `torch.addmm()` for matrix multiplication accumulation

### Basic Helion Pattern

```python
import helion
import helion.language as hl
import torch

@helion.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()

    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)

    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc

    return out
```

### Helion vs Triton Key Differences

| Aspect | Helion | Triton |
|--------|--------|--------|
| Tiling | Automatic with `hl.tile()` | Manual with `tl.arange()` |
| Memory | Implicit loads/stores | Explicit `tl.load/store` |
| Types | Use PyTorch types | Use Triton types (`tl.float32`) |
| Indexing | Use Python slicing | Use pointer arithmetic |

### Best Practices

1. **Use `static_shapes=True`** for matmul operations to enable better optimization
2. **Use `torch.float32` for accumulators** to maintain numerical precision
3. **Use `torch.addmm`** for matrix multiplication accumulation (preferred pattern)
4. **Declare output tensors at the top** before any loops

### Common Patterns

**Element-wise Operations:**
```python
@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return out
```

**Reduction Operations:**
```python
@helion.kernel()
def sum_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            acc = acc + x[tile_m, tile_n].sum(dim=-1)
        out[tile_m] = acc.to(x.dtype)
    return out
```
"""
