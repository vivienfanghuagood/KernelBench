import os
import sys
import asyncio
import traceback
from typing import Dict, Any, Optional

# Add the parent directory to the path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.prompt_constructor_multilang import get_prompt_for_backend
from src.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    set_gpu_arch,
)
from api.database import db, GenerationStatus

class KernelGenerationService:
    def __init__(self):
        self.repo_top_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    async def generate_kernel_async(self, request_id: str):
        """Asynchronously generate a kernel for the given request"""
        try:
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
            
            # Generate kernel - run in thread pool since it's blocking
            loop = asyncio.get_event_loop()
            custom_kernel = await loop.run_in_executor(None, inference_server, custom_prompt)
            # import pdb;pdb.set_trace()
            custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])
            
            if custom_kernel is None:
                raise ValueError(f"Custom {backend} kernel code generation failed")
            
            # Evaluate kernel (optional, can be made configurable)
            try:
                eval_result = await loop.run_in_executor(
                    None,
                    self._evaluate_kernel,
                    ref_arch_src,
                    custom_kernel,
                    backend
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
    
    def _evaluate_kernel(self, ref_arch_src: str, custom_kernel: str, backend: str):
        """Evaluate the generated kernel against reference"""
        return eval_kernel_against_ref(
            ref_arch_src,
            custom_kernel,
            verbose=False,
            measure_performance=True,
            num_correct_trials=3,  # Reduced for faster API response
            num_perf_trials=50,    # Reduced for faster API response
            backend=backend,
        )
    
    def submit_generation_request(self, 
                                ref_arch_src: str,
                                gpu_arch: list,
                                backend: str,
                                model_name: str,
                                server_type: str,
                                max_tokens: int = 4096,
                                temperature: float = 0.0) -> str:
        """Submit a new kernel generation request"""
        request_id = db.create_generation_request(
            ref_arch_src=ref_arch_src,
            gpu_arch=gpu_arch,
            backend=backend,
            model_name=model_name,
            server_type=server_type,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Start generation in background
        asyncio.create_task(self.generate_kernel_async(request_id))
        
        return request_id
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get the status and results of a generation request"""
        return db.get_request(request_id)
    
    def get_all_requests(self, limit: int = 100) -> list:
        """Get all generation requests"""
        return db.get_all_requests(limit)

# Global service instance
kernel_service = KernelGenerationService()