"""
Benchmark Script for Batch Generation
Sequentially calls the backend generate API for all datasets in level1 and level2.
Records Status, Speedup, Mean Runtime, and other metrics for each generation.
"""

import os
import json
import time
import re
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import csv
import argparse

CUSTOM_PROMPT = """
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
class BenchmarkRunner:
    def __init__(self, api_base_url: str = "http://localhost:8009"):
        self.api_base_url = api_base_url
        self.results: List[Dict[str, Any]] = []
        
    def load_sample_files(self) -> List[Dict[str, str]]:
        """Load all sample files from level1 and level2"""
        samples = []
        base_path = Path("KernelBench")
        
        for level in ["level1", "level2"]:
            level_path = base_path / level
            if level_path.exists():
                files = sorted([f for f in level_path.iterdir() if f.suffix == '.py'])
                for file_path in files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    samples.append({
                        "name": file_path.stem,
                        "filename": file_path.name,
                        "level": level,
                        "path": str(file_path),
                        "content": content
                    })
        
        return samples
    
    def submit_generation_request(
        self,
        ref_arch_src: str,
        backend: str = "triton",
        gpu_arch: str = "4090",
        model_name: str = "gpt-5",
        server_type: str = "openai",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        custom_prompt: Optional[str] = None,
        problem_name: Optional[str] = None,
        max_retries: int = 3
    ) -> Optional[str]:
        """Submit a generation request and return request_id"""
        try:
            request_data = {
                "ref_arch_src": ref_arch_src,
                "gpu_arch": [gpu_arch],
                "backend": backend,
                "model_name": model_name,
                "server_type": server_type,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "custom_prompt": custom_prompt if custom_prompt and custom_prompt.strip() else None,
                "problem_name": problem_name if problem_name and problem_name.strip() else None,
                "max_retries": max_retries
            }
            
            response = requests.post(
                f"{self.api_base_url}/api/generate",
                json=request_data,
                timeout=30
            )
            
            if not response.ok:
                error_detail = response.json().get("detail", "Unknown error")
                print(f"❌ Failed to submit request: {error_detail}")
                return None
            
            result = response.json()
            return result.get("request_id")
            
        except Exception as e:
            print(f"❌ Error submitting request: {str(e)}")
            return None
    
    def wait_for_completion(
        self, 
        request_id: str, 
        max_wait_time: int = 600,
        poll_interval: int = 20
    ) -> Optional[Dict[str, Any]]:
        """Wait for a request to complete and return the status data"""
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > max_wait_time:
                print(f"⏱️ Request timed out after {max_wait_time}s")
                return None
            
            try:
                response = requests.get(
                    f"{self.api_base_url}/api/status/{request_id}",
                    timeout=10
                )
                
                if not response.ok:
                    print(f"❌ Failed to check status")
                    return None
                
                status_data = response.json()
                current_status = status_data.get("status")
                
                if current_status == "pending":
                    print(f"⏳ Queued... ({int(elapsed)}s elapsed)")
                elif current_status == "processing":
                    print(f"🔄 Generating... ({int(elapsed)}s elapsed)")
                elif current_status == "completed":
                    print(f"✅ Completed in {int(elapsed)}s")
                    return status_data
                elif current_status == "failed":
                    error_msg = status_data.get("error_message", "Unknown error")
                    print(f"❌ Failed: {error_msg}")
                    return status_data
                
                time.sleep(poll_interval)
                
            except Exception as e:
                print(f"❌ Error checking status: {str(e)}")
                time.sleep(poll_interval)
    
    def parse_eval_string(self, s: str) -> Dict[str, Any]:
        """Parse evaluation result string (adapted from gradio_app.py)"""
        result = {
            "compiled": None,
            "correctness": None,
            "metadata": {},
            "runtime": None,
            "runtime_stats": {},
            "ref_runtime": None,
            "ref_runtime_stats": {},
            "speedup": None
        }
        
        if not s:
            return result
        
        try:
            # Parse compiled status
            compiled_match = re.search(r'compiled=(True|False)', s)
            if compiled_match:
                result["compiled"] = compiled_match.group(1) == "True"
            
            # Parse correctness status
            correctness_match = re.search(r'correctness=(True|False)', s)
            if correctness_match:
                result["correctness"] = correctness_match.group(1) == "True"
            
            # Parse metadata
            metadata_match = re.search(r'metadata=(\{[^}]*\})', s)
            if metadata_match:
                try:
                    metadata_str = metadata_match.group(1).replace("'", '"')
                    metadata_str = re.sub(r'\((\d+)\s*/\s*(\d+)\)', r'"(\1 / \2)"', metadata_str)
                    result["metadata"] = json.loads(metadata_str)
                except:
                    pass
            
            # Parse runtime
            runtime_match = re.search(r'\bruntime=([\d.]+)(?=\s|$|runtime_stats|ref_)', s)
            if runtime_match:
                result["runtime"] = float(runtime_match.group(1))
            
            # Parse ref_runtime
            ref_runtime_match = re.search(r'ref_runtime=([\d.]+)', s)
            if ref_runtime_match:
                result["ref_runtime"] = float(ref_runtime_match.group(1))
            
            # Parse speedup
            speedup_match = re.search(r'speedup=([\d.]+)', s)
            if speedup_match:
                result["speedup"] = float(speedup_match.group(1))
            
            # Extract dictionary values
            def extract_dict(start_str: str) -> Optional[Dict]:
                start_idx = s.find(start_str)
                if start_idx == -1:
                    return None
                
                start_pos = start_idx + len(start_str)
                brace_count = 0
                end_pos = start_pos
                
                for i in range(start_pos, len(s)):
                    if s[i] == '{':
                        brace_count += 1
                    elif s[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > start_pos:
                    dict_str = s[start_pos:end_pos]
                    dict_str = dict_str.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
                    try:
                        return json.loads(dict_str)
                    except:
                        pass
                return None
            
            # Parse runtime_stats
            runtime_stats = extract_dict('runtime_stats=')
            if runtime_stats:
                result["runtime_stats"] = runtime_stats
            
            # Parse ref_runtime_stats
            ref_runtime_stats = extract_dict('ref_runtime_stats=')
            if ref_runtime_stats:
                result["ref_runtime_stats"] = ref_runtime_stats
                
        except Exception as e:
            print(f"Warning: Error parsing eval string: {e}")
        
        return result
    
    def process_sample(
        self,
        sample: Dict[str, str],
        backend: str = "triton",
        gpu_arch: str = "4090",
        model_name: str = "gpt-5",
        server_type: str = "openai",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        custom_prompt: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Process a single sample and return the results"""
        print(f"\n{'='*80}")
        print(f"Processing: [{sample['level']}] {sample['name']}")
        print(f"{'='*80}")
        
        result = {
            "sample_name": sample['name'],
            "filename": sample['filename'],
            "level": sample['level'],
            "backend": backend,
            "gpu_arch": gpu_arch,
            "model_name": model_name,
            "server_type": server_type,
            "custom_prompt": custom_prompt,
            "timestamp": datetime.now().isoformat(),
            "request_id": None,
            "status": "failed",
            "compiled": None,
            "correctness": None,
            "runtime": None,
            "runtime_mean": None,
            "runtime_std": None,
            "runtime_min": None,
            "runtime_max": None,
            "ref_runtime": None,
            "ref_runtime_mean": None,
            "ref_runtime_std": None,
            "ref_runtime_min": None,
            "ref_runtime_max": None,
            "speedup": None,
            "num_trials": None,
            "hardware": None,
            "device": None,
            "error_message": None,
            "total_time_seconds": 0
        }
        
        start_time = time.time()
        
        # Submit request (use sample name as problem_name)
        request_id = self.submit_generation_request(
            ref_arch_src=sample['content'],
            backend=backend,
            gpu_arch=gpu_arch,
            model_name=model_name,
            server_type=server_type,
            max_tokens=max_tokens,
            temperature=temperature,
            custom_prompt=custom_prompt,
            problem_name=sample['name'],  # Use sample name as problem_name
            max_retries=max_retries
        )
        
        if not request_id:
            result["error_message"] = "Failed to submit request"
            result["total_time_seconds"] = time.time() - start_time
            return result
        
        result["request_id"] = request_id
        print(f"📝 Request ID: {request_id[:12]}...")
        
        # Wait for completion
        status_data = self.wait_for_completion(request_id)
        
        if not status_data:
            result["error_message"] = "Request timed out or failed"
            result["total_time_seconds"] = time.time() - start_time
            return result
        
        # Update basic status
        result["status"] = status_data.get("status", "unknown")
        result["error_message"] = status_data.get("error_message")
        
        # Parse evaluation results
        eval_result_str = status_data.get("eval_result", "")
        if eval_result_str:
            eval_data = self.parse_eval_string(eval_result_str)
            
            result["compiled"] = eval_data.get("compiled")
            result["correctness"] = eval_data.get("correctness")
            result["runtime"] = eval_data.get("runtime")
            result["ref_runtime"] = eval_data.get("ref_runtime")
            result["speedup"] = eval_data.get("speedup")
            
            # Runtime stats
            runtime_stats = eval_data.get("runtime_stats", {})
            if runtime_stats:
                result["runtime_mean"] = runtime_stats.get("mean")
                result["runtime_std"] = runtime_stats.get("std")
                result["runtime_min"] = runtime_stats.get("min")
                result["runtime_max"] = runtime_stats.get("max")
                result["num_trials"] = runtime_stats.get("num_trials")
            
            # Ref runtime stats
            ref_runtime_stats = eval_data.get("ref_runtime_stats", {})
            if ref_runtime_stats:
                result["ref_runtime_mean"] = ref_runtime_stats.get("mean")
                result["ref_runtime_std"] = ref_runtime_stats.get("std")
                result["ref_runtime_min"] = ref_runtime_stats.get("min")
                result["ref_runtime_max"] = ref_runtime_stats.get("max")
            
            # Metadata
            metadata = eval_data.get("metadata", {})
            if metadata:
                result["hardware"] = metadata.get("hardware")
                result["device"] = metadata.get("device")
        
        result["total_time_seconds"] = time.time() - start_time
        
        # Print summary
        print(f"\n📊 Results Summary:")
        print(f"   Status: {result['status']}")
        print(f"   Compiled: {result['compiled']}")
        print(f"   Correctness: {result['correctness']}")
        if result['speedup'] is not None:
            print(f"   Speedup: {result['speedup']:.2f}x")
        if result['runtime'] is not None:
            print(f"   Runtime: {result['runtime']:.2f} ms")
        if result['ref_runtime'] is not None:
            print(f"   Ref Runtime: {result['ref_runtime']:.2f} ms")
        print(f"   Total Time: {result['total_time_seconds']:.1f}s")
        
        return result
    
    def run_benchmark(
        self,
        backend: str = "triton",
        gpu_arch: str = "4090",
        model_name: str = "gpt-5",
        server_type: str = "openai",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        custom_prompt: Optional[str] = None,
        output_dir: str = "results/benchmark",
        start_from: Optional[str] = None,
        limit: Optional[int] = None,
        max_retries: int = 3
    ):
        """Run benchmark on all samples"""
        print("🚀 Starting Benchmark Run")
        print(f"   Backend: {backend}")
        print(f"   GPU Arch: {gpu_arch}")
        print(f"   Model: {model_name}")
        print(f"   Server: {server_type}")
        if custom_prompt:
            print(f"   Custom Prompt: {custom_prompt[:50]}..." if len(custom_prompt) > 50 else f"   Custom Prompt: {custom_prompt}")
        
        # Load all samples
        print("\n📂 Loading samples...")
        samples = self.load_sample_files()
        print(f"   Found {len(samples)} samples")
        
        # Filter samples if start_from is specified
        if start_from:
            start_idx = next((i for i, s in enumerate(samples) if s['name'] == start_from), None)
            if start_idx is not None:
                samples = samples[start_idx:]
                print(f"   Starting from: {start_from} (index {start_idx})")
            else:
                print(f"   Warning: start_from '{start_from}' not found, processing all samples")
        
        # Limit number of samples if specified
        if limit:
            samples = samples[:limit]
            print(f"   Limited to first {limit} samples")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_path / f"benchmark_{backend}_{model_name}_{timestamp}.json"
        csv_file = output_path / f"benchmark_{backend}_{model_name}_{timestamp}.csv"
        
        # Process each sample sequentially
        total_samples = len(samples)
        for idx, sample in enumerate(samples, 1):
            print(f"\n{'#'*80}")
            print(f"# Progress: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
            print(f"{'#'*80}")
            # import pdb;pdb.set_trace()
            if sample["name"].find("conv") != -1 or sample["name"].find("Conv") != -1 or sample["name"].find("cumsom") != -1 or sample["name"].find("Loss") != -1:
                continue
            
            result = self.process_sample(
                sample=sample,
                backend=backend,
                gpu_arch=gpu_arch,
                model_name=model_name,
                server_type=server_type,
                max_tokens=max_tokens,
                temperature=temperature,
                custom_prompt=custom_prompt,
                max_retries=max_retries
            )
            
            self.results.append(result)
            
            # Save results after each sample (incremental save)
            self.save_results(results_file, csv_file)
            
            # Small delay between requests
            if idx < total_samples:
                print("\n⏸️ Waiting 5 seconds before next request...")
                time.sleep(5)
        
        # Print final summary
        self.print_summary()
        
        print(f"\n✅ Benchmark completed!")
        print(f"   Results saved to: {results_file}")
        print(f"   CSV saved to: {csv_file}")
    
    def save_results(self, json_file: Path, csv_file: Path):
        """Save results to JSON and CSV files"""
        # Save JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save CSV
        if self.results:
            fieldnames = [
                "sample_name", "filename", "level", "backend", "gpu_arch", 
                "model_name", "server_type", "custom_prompt", "timestamp", "request_id",
                "status", "compiled", "correctness",
                "runtime", "runtime_mean", "runtime_std", "runtime_min", "runtime_max",
                "ref_runtime", "ref_runtime_mean", "ref_runtime_std", "ref_runtime_min", "ref_runtime_max",
                "speedup", "num_trials", "hardware", "device",
                "error_message", "total_time_seconds"
            ]
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.results)
    
    def print_summary(self):
        """Print benchmark summary statistics"""
        if not self.results:
            return
        
        total = len(self.results)
        completed = sum(1 for r in self.results if r['status'] == 'completed')
        failed = sum(1 for r in self.results if r['status'] == 'failed')
        compiled = sum(1 for r in self.results if r['compiled'] is True)
        correct = sum(1 for r in self.results if r['correctness'] is True)
        
        speedups = [r['speedup'] for r in self.results if r['speedup'] is not None]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        
        total_time = sum(r['total_time_seconds'] for r in self.results)
        
        print(f"\n{'='*80}")
        print("📈 BENCHMARK SUMMARY")
        print(f"{'='*80}")
        print(f"Total Samples: {total}")
        print(f"Completed: {completed} ({completed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Compiled: {compiled} ({compiled/total*100:.1f}%)")
        print(f"Correct: {correct} ({correct/total*100:.1f}%)")
        if speedups:
            print(f"Average Speedup: {avg_speedup:.2f}x")
            print(f"Max Speedup: {max(speedups):.2f}x")
            print(f"Min Speedup: {min(speedups):.2f}x")
        print(f"Total Time: {total_time/60:.1f} minutes")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark KernelBench Generation API")
    parser.add_argument("--api-url", type=str, default="http://localhost:8009",
                        help="API base URL (default: http://localhost:8009)")
    parser.add_argument("--backend", type=str, default="triton",
                        choices=["cuda", "triton", "cute"],
                        help="Backend to use (default: triton)")
    parser.add_argument("--gpu-arch", type=str, default="4090",
                        help="GPU architecture (default: 4090)")
    parser.add_argument("--model-name", type=str, default="gpt-5",
                        help="Model name (default: gpt-5)")
    parser.add_argument("--server-type", type=str, default="openai",
                        choices=["openai", "anthropic", "google", "deepseek"],
                        help="Server type (default: openai)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max tokens (default: 4096)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature (default: 0.0)")
    parser.add_argument("--custom-prompt", type=str, default=CUSTOM_PROMPT,
                        help="Custom prompt to append to generation prompt (default: None)")
    parser.add_argument("--output-dir", type=str, default="results/benchmark",
                        help="Output directory (default: results/benchmark)")
    parser.add_argument("--start-from", type=str, default=None,
                        help="Start from specific sample name (e.g., '50_conv_standard_2D__square_input__square_kernel')")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries on failure (default: 3)")
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(api_base_url=args.api_url)
    
    try:
        runner.run_benchmark(
            backend=args.backend,
            gpu_arch=args.gpu_arch,
            model_name=args.model_name,
            server_type=args.server_type,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            custom_prompt=args.custom_prompt,
            output_dir=args.output_dir,
            start_from=args.start_from,
            limit=args.limit,
            max_retries=args.max_retries
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
        print("Saving partial results...")
        
        if runner.results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / f"benchmark_{args.backend}_{args.model_name}_{timestamp}_partial.json"
            csv_file = output_path / f"benchmark_{args.backend}_{args.model_name}_{timestamp}_partial.csv"
            
            runner.save_results(results_file, csv_file)
            runner.print_summary()
            
            print(f"   Partial results saved to: {results_file}")
            print(f"   Partial CSV saved to: {csv_file}")
    except Exception as e:
        print(f"\n❌ Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
