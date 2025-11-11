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
            custom_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
        elif backend in ["triton", "cute"]:
            custom_prompt = get_prompt_for_backend(ref_arch_src, backend)
        else:
            raise ValueError(f"Unsupported backend: {backend}. Must be 'cuda', 'triton', or 'cute'.")
        
        # Generate kernel
        custom_kernel = inference_server(custom_prompt)
        custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])
        
        if custom_kernel is None:
            raise ValueError(f"Custom {backend} kernel code generation failed")
        
        # Evaluate kernel (optional, can be made configurable)
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
            eval_result_str = str(eval_result)
        except Exception as eval_error:
            eval_result_str = f"Evaluation failed: {str(eval_error)}"
        
        # Update request with results
        db.update_request_status(
            request_id, 
            GenerationStatus.COMPLETED,
            generated_kernel=custom_kernel,
            eval_result=eval_result_str
        )
        
    except Exception as e:
        error_message = f"Generation failed: {str(e)}\n{traceback.format_exc()}"
        db.update_request_status(
            request_id,
            GenerationStatus.FAILED,
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
                                temperature: float = 0.0) -> str:
        """Submit a new kernel generation request using multiprocessing"""
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
        
        # Create request in database
        request_id = db.create_generation_request(
            ref_arch_src=ref_arch_src,
            gpu_arch=gpu_arch,
            backend=backend,
            model_name=model_name,
            server_type=server_type,
            max_tokens=max_tokens,
            temperature=temperature
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