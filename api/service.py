import os
import sys
import traceback
import multiprocessing as mp
import signal
from typing import Dict, Any, Optional
from datetime import datetime

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import APIConfig

from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.prompt_constructor_multilang import get_prompt_for_backend
from src.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    set_gpu_arch,
)
from api.database import db, GenerationStatus

def _save_prompt_to_file(request_id: str, attempt: int, prompt: str):
    """Save the prompt for a generation attempt to a log file
    
    Args:
        request_id: The unique request identifier
        attempt: The attempt number (0-indexed)
        prompt: The prompt text to save
    """
    try:
        # Ensure logs directory exists
        logs_dir = APIConfig.LOGS_DIR
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create filename with request_id and trial number
        filename = f"{request_id}_trial{attempt}.txt"
        filepath = os.path.join(logs_dir, filename)
        
        # Write prompt to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Request ID: {request_id}\n")
            f.write(f"Attempt: {attempt}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(prompt)
    except Exception as e:
        # Log error but don't fail the generation process
        print(f"Warning: Failed to save prompt to file: {e}")

def _construct_reflection_prompt(original_prompt: str, generated_code: str, error_message: str, retry_count: int) -> str:
    """Construct a reflection prompt with error feedback for LLM to retry generation
    
    Args:
        original_prompt: The original prompt used for generation
        generated_code: The code that was generated but failed
        error_message: The error message from compilation or correctness check
        retry_count: Current retry attempt number
    
    Returns:
        A new prompt that includes the original request, previous attempt, and error feedback
    """
    reflection_prompt = f"""The previous attempt to generate the kernel failed. Here is the feedback:

ORIGINAL REQUEST:
{original_prompt}

PREVIOUS GENERATED CODE (Attempt #{retry_count}):
```
{generated_code}
```

ERROR MESSAGE:
{error_message}

Please analyze the error and generate a corrected version of the kernel that fixes the issues mentioned above. Make sure to:
1. Carefully read the error message and understand what went wrong
2. Fix the compilation errors or correctness issues
3. Generate complete, working code that passes all tests
4. Follow the same requirements from the original request

Generate the corrected kernel code:"""
    
    return reflection_prompt

def _set_process_limits():
    """Set resource limits for worker processes"""
    try:
        import resource
        # Limit core dump file size
        core_size = APIConfig.CORE_DUMP_MAX_SIZE_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CORE, (core_size, core_size))
        
        # Set nice priority (lower priority to not interfere with main process)
        os.nice(5)
    except (ImportError, AttributeError, OSError):
        # resource module not available on Windows or operation not permitted
        pass

def _worker_generate_kernel(request_id: str, repo_top_dir: str):
    """Worker function to generate kernel in a separate process"""
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        db.update_request_status(
            request_id,
            GenerationStatus.FAILED,
            error_message=f"Worker process terminated by signal {signum}"
        )
        sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGBREAK'):  # Windows
        signal.signal(signal.SIGBREAK, signal_handler)
    
    # Track custom_kernel at outer scope for exception handling
    custom_kernel = None
    final_eval_result_str = None
    
    try:
        # Set resource limits
        _set_process_limits()
        
        # Update status to processing
        db.update_request_status(request_id, GenerationStatus.PROCESSING)
        
        # Get request details
        request_data = db.get_request(request_id)
        if not request_data:
            raise ValueError(f"Request {request_id} not found")
        
        # Extract parameters
        ref_arch_src = request_data['ref_arch_src']
        gpu_arch = request_data['gpu_arch']
        backend = request_data['backend']
        model_name = request_data['model_name']
        server_type = request_data['server_type']
        max_tokens = request_data['max_tokens']
        temperature = request_data['temperature']
        custom_prompt = request_data.get('custom_prompt', None)
        problem_name = request_data.get('problem_name', None)
        max_retries = request_data.get('max_retries', APIConfig.DEFAULT_MAX_RETRIES)
        current_retry = request_data.get('current_retry', 0)
        retry_history = request_data.get('retry_history', [])
        
        # Set GPU architecture
        if gpu_arch:
            set_gpu_arch(gpu_arch)
        
        # Create inference server
        inference_server = create_inference_server_from_presets(
            server_type=server_type,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=False,
            time_generation=True,
        )
        
        # Generate prompt based on backend
        if backend == "cuda":
            original_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
        elif backend in ["triton", "cute"]:
            original_prompt = get_prompt_for_backend(ref_arch_src, backend)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Must be 'cuda', 'triton', or 'cute'.")
        
        # Append custom prompt if provided
        if custom_prompt and custom_prompt.strip():
            original_prompt = original_prompt + "\n\n" + custom_prompt.strip()
        
        # Retry loop with reflection
        # custom_kernel = None  # Already declared at outer scope
        eval_result = None
        eval_error_msg = None
        # final_eval_result_str = None  # Already declared at outer scope
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            current_retry = attempt
            
            # Determine prompt to use
            if attempt == 0:
                # First attempt: use original prompt
                prompt_to_use = original_prompt
            else:
                # Retry attempt: construct reflection prompt with error feedback
                prompt_to_use = _construct_reflection_prompt(
                    original_prompt, 
                    custom_kernel, 
                    eval_error_msg, 
                    attempt
                )
            
            # Save prompt to log file
            _save_prompt_to_file(request_id, attempt, prompt_to_use)
            
            # Generate kernel
            try:
                custom_kernel = inference_server(prompt_to_use)
                custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])
                
                if custom_kernel is None:
                    eval_error_msg = f"Custom {backend} kernel code generation failed - no valid code extracted"
                    # Record this attempt in history
                    retry_history.append({
                        'attempt': attempt,
                        'generated_code': None,
                        'error': eval_error_msg
                    })
                    # Update database with current retry status
                    db.update_request_status(
                        request_id,
                        GenerationStatus.PROCESSING,
                        current_retry=current_retry,
                        retry_history=retry_history
                    )
                    continue
            except Exception as gen_error:
                eval_error_msg = f"Generation error: {str(gen_error)}\n{traceback.format_exc()}"
                retry_history.append({
                    'attempt': attempt,
                    'generated_code': None,
                    'error': eval_error_msg
                })
                db.update_request_status(
                    request_id,
                    GenerationStatus.PROCESSING,
                    current_retry=current_retry,
                    retry_history=retry_history
                )
                continue
            
            # Evaluate kernel
            eval_error_msg = None
            try:
                eval_result = eval_kernel_against_ref(
                    ref_arch_src,
                    custom_kernel,
                    verbose=False,
                    measure_performance=True,
                    num_correct_trials=APIConfig.DEFAULT_NUM_CORRECT_TRIALS,
                    num_perf_trials=APIConfig.DEFAULT_NUM_PERF_TRIALS,
                    backend=backend,
                )
                final_eval_result_str = str(eval_result)
                
                # Check if generation was successful
                # Success criteria: either correctness=True OR compiled=True (even if incorrect)
                # Only retry if compilation failed
                if eval_result.correctness or eval_result.compiled:
                    # Success! Either fully correct or at least compiled successfully
                    # Mark as completed even if correctness=False but compiled=True
                    retry_history.append({
                        'attempt': attempt,
                        'generated_code': custom_kernel[:500] + '...' if len(custom_kernel) > 500 else custom_kernel,
                        'success': True,
                        'eval_result': final_eval_result_str
                    })
                    db.update_request_status(
                        request_id, 
                        GenerationStatus.COMPLETED,
                        generated_kernel=custom_kernel,
                        eval_result=final_eval_result_str,
                        current_retry=current_retry,
                        retry_history=retry_history
                    )
                    return  # Exit successfully
                else:
                    # Only retry if compilation failed (compiled=False)
                    # Extract error message for reflection
                    if 'runtime_error_traceback' in eval_result.metadata:
                        eval_error_msg = str(eval_result.metadata['runtime_error_traceback'])
                    elif 'runtime_error' in eval_result.metadata:
                        eval_error_msg = str(eval_result.metadata['runtime_error'])
                    elif 'compilation_error' in eval_result.metadata:
                        eval_error_msg = str(eval_result.metadata['compilation_error'])
                    else:
                        eval_error_msg = f"Correctness check failed. Metadata: {str(eval_result.metadata)}"
                    
                    # Record this attempt in history
                    retry_history.append({
                        'attempt': attempt,
                        'generated_code': custom_kernel[:500] + '...' if len(custom_kernel) > 500 else custom_kernel,
                        'error': eval_error_msg,
                        'success': False
                    })
                    
                    # Update database with current retry status
                    db.update_request_status(
                        request_id,
                        GenerationStatus.PROCESSING,
                        current_retry=current_retry,
                        retry_history=retry_history
                    )
                    
            except Exception as eval_error:
                eval_error_msg = f"Evaluation failed: {str(eval_error)}\n{traceback.format_exc()}"
                final_eval_result_str = eval_error_msg
                
                # Record this attempt in history
                retry_history.append({
                    'attempt': attempt,
                    'generated_code': custom_kernel[:500] + '...' if len(custom_kernel) > 500 else custom_kernel,
                    'error': eval_error_msg,
                    'success': False
                })
                
                # Update database with current retry status
                db.update_request_status(
                    request_id,
                    GenerationStatus.PROCESSING,
                    current_retry=current_retry,
                    retry_history=retry_history
                )
        
        # If we reach here, all retries have been exhausted without successful compilation
        # This means all attempts failed to compile (compiled=False in all attempts)
        error_summary = f"All {max_retries + 1} compilation attempts failed. Last error: {eval_error_msg}"
        
        # Save final state with reference code, generated kernel, and error message
        # ref_arch_src is already stored in the database from creation
        # We explicitly save the last generated_kernel and comprehensive error info
        db.update_request_status(
            request_id,
            GenerationStatus.FAILED,
            generated_kernel=custom_kernel,  # Last attempt's generated code
            eval_result=final_eval_result_str,
            error_message=error_summary,
            current_retry=current_retry,
            retry_history=retry_history
        )
        
    except Exception as e:
        # Catch-all for unexpected errors - try to save whatever we have
        error_message = f"Generation failed: {str(e)}\n{traceback.format_exc()}"
        db.update_request_status(
            request_id,
            GenerationStatus.FAILED,
            generated_kernel=custom_kernel,  # May be None or partial result
            eval_result=final_eval_result_str,  # May be None
            error_message=error_message
        )

class KernelGenerationService:
    def __init__(self, max_workers: Optional[int] = None):
        self.repo_top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.max_workers = max_workers or APIConfig.MAX_WORKERS
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set
            pass
    
    def _cleanup_finished_processes(self):
        """Remove finished processes from active_processes"""
        finished = []
        for rid, proc_info in self.active_processes.items():
            proc = proc_info['process']
            if not proc.is_alive():
                finished.append(rid)
                proc.join(timeout=1)
                if proc.exitcode is None:
                    proc.terminate()
                elif proc.exitcode != 0:
                    # Process crashed, update status if not already updated
                    request = db.get_request(rid)
                    if request and request['status'] == GenerationStatus.PROCESSING.value:
                        db.update_request_status(
                            rid,
                            GenerationStatus.FAILED,
                            error_message=f"Worker process crashed with exit code {proc.exitcode}"
                        )
        
        for rid in finished:
            self.active_processes.pop(rid, None)
    
    def _check_process_timeout(self, request_id: str) -> bool:
        """Check if a process has exceeded timeout"""
        proc_info = self.active_processes.get(request_id)
        if not proc_info:
            return False
        
        start_time = proc_info['start_time']
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if elapsed > APIConfig.WORKER_TIMEOUT:
            proc = proc_info['process']
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=APIConfig.PROCESS_TERM_TIMEOUT)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            
            db.update_request_status(
                request_id,
                GenerationStatus.FAILED,
                error_message=f"Request timed out after {APIConfig.WORKER_TIMEOUT} seconds"
            )
            self.active_processes.pop(request_id, None)
            return True
        
        return False
    
    def submit_generation_request(self, 
                                ref_arch_src: str,
                                gpu_arch: list,
                                backend: str,
                                model_name: str,
                                server_type: str,
                                max_tokens: int = 4096,
                                temperature: float = 0.0,
                                custom_prompt: str = None,
                                problem_name: str = None,
                                max_retries: int = None) -> str:
        """Submit a new kernel generation request using multiprocessing
        
        Args:
            ref_arch_src: Reference architecture source code
            gpu_arch: GPU architecture specification
            backend: Backend type (cuda, triton, cute)
            model_name: Name of the LLM model to use
            server_type: Type of inference server
            max_tokens: Maximum tokens for generation
            temperature: Temperature for sampling
            custom_prompt: Optional custom prompt to append
            problem_name: Optional problem name
            max_retries: Maximum number of retries on failure (default: APIConfig.DEFAULT_MAX_RETRIES)
        
        Returns:
            request_id: Unique identifier for the generation request
        """
        # Clean up finished processes and check timeouts
        self._cleanup_finished_processes()
        for request_id in list(self.active_processes.keys()):
            self._check_process_timeout(request_id)
        
        # Check if we have reached max workers
        if len(self.active_processes) >= self.max_workers:
            raise RuntimeError(
                f"Maximum number of concurrent workers ({self.max_workers}) reached. "
                f"Please wait for some tasks to complete."
            )
        
        # Use default max_retries if not provided
        if max_retries is None:
            max_retries = APIConfig.DEFAULT_MAX_RETRIES
        
        # Create request in database
        request_id = db.create_generation_request(
            ref_arch_src=ref_arch_src,
            gpu_arch=gpu_arch,
            backend=backend,
            model_name=model_name,
            server_type=server_type,
            max_tokens=max_tokens,
            temperature=temperature,
            custom_prompt=custom_prompt,
            problem_name=problem_name,
            max_retries=max_retries
        )
        
        # Start generation in separate process
        process = mp.Process(
            target=_worker_generate_kernel,
            args=(request_id, self.repo_top_dir),
            daemon=False  # Don't make it daemon to ensure cleanup
        )
        process.start()
        self.active_processes[request_id] = {
            'process': process,
            'start_time': datetime.now()
        }
        
        return request_id
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status and results of a generation request"""
        return db.get_request(request_id)
    
    def get_all_requests(self, limit: int = 100) -> list:
        """Get all generation requests"""
        return db.get_all_requests(limit)
    
    def terminate_request(self, request_id: str) -> bool:
        """Terminate a running generation request"""
        if request_id in self.active_processes:
            proc_info = self.active_processes[request_id]
            proc = proc_info['process']
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=APIConfig.PROCESS_TERM_TIMEOUT)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            self.active_processes.pop(request_id)
            db.update_request_status(
                request_id,
                GenerationStatus.FAILED,
                error_message="Request terminated by user"
            )
            return True
        return False
    
    def cleanup_all_processes(self):
        """Cleanup all active processes - call this on shutdown"""
        for request_id, proc_info in list(self.active_processes.items()):
            proc = proc_info['process']
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=APIConfig.PROCESS_TERM_TIMEOUT)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
        self.active_processes.clear()
    
    def get_active_workers_count(self) -> int:
        """Get the number of currently active worker processes"""
        self._cleanup_finished_processes()
        return len(self.active_processes)

# Global service instance
kernel_service = KernelGenerationService()