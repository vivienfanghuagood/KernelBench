"""
Helpers for Evaluations

Supports both NVIDIA CUDA and AMD ROCm GPUs.
ROCm support is provided through PyTorch's HIP backend, which exposes
the same torch.cuda API for AMD GPUs.
"""
import hashlib
import importlib
import json
import linecache
import os, subprocess
import random
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Union, Literal

"Use Literal to specify the type of the device"

import numpy as np
import requests
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict

from . import utils


################################################################################
# GPU Detection and Compatibility
################################################################################

def is_rocm_available() -> bool:
    """
    Check if ROCm (AMD GPU) is available.
    ROCm uses PyTorch's HIP backend which exposes torch.cuda API.
    """
    if not torch.cuda.is_available():
        return False
    # Check for HIP version (ROCm indicator)
    return hasattr(torch.version, 'hip') and torch.version.hip is not None


def is_cuda_available() -> bool:
    """
    Check if NVIDIA CUDA is available (not ROCm).
    """
    if not torch.cuda.is_available():
        return False
    return not is_rocm_available()


def get_gpu_vendor() -> Literal["nvidia", "amd", "unknown"]:
    """
    Detect the GPU vendor (NVIDIA or AMD).
    """
    if not torch.cuda.is_available():
        return "unknown"
    if is_rocm_available():
        return "amd"
    return "nvidia"


def get_gpu_info(device: torch.device = None) -> dict:
    """
    Get GPU information including vendor, name, and memory.
    
    Returns:
        dict with keys: vendor, name, memory_total_gb, compute_capability (NVIDIA only)
    """
    if device is None:
        device = torch.cuda.current_device()
    
    info = {
        "vendor": get_gpu_vendor(),
        "name": torch.cuda.get_device_name(device),
        "memory_total_gb": torch.cuda.get_device_properties(device).total_memory / (1024**3),
    }
    
    # Add compute capability for NVIDIA GPUs
    if info["vendor"] == "nvidia":
        props = torch.cuda.get_device_properties(device)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    
    # Add ROCm-specific info for AMD GPUs
    if info["vendor"] == "amd":
        info["hip_version"] = torch.version.hip
        # Try to get architecture info
        try:
            props = torch.cuda.get_device_properties(device)
            info["gcn_arch"] = getattr(props, 'gcnArchName', 'unknown')
        except:
            pass
    
    return info


def check_gpu_available(verbose: bool = False) -> bool:
    """
    Check if any GPU (CUDA or ROCm) is available.
    
    Args:
        verbose: If True, print GPU information
    
    Returns:
        True if GPU is available, False otherwise
    """
    if not torch.cuda.is_available():
        if verbose:
            print("[GPU] No GPU available")
        return False
    
    if verbose:
        gpu_info = get_gpu_info()
        vendor_name = "AMD ROCm" if gpu_info["vendor"] == "amd" else "NVIDIA CUDA"
        print(f"[GPU] {vendor_name} available: {gpu_info['name']}")
        print(f"[GPU] Memory: {gpu_info['memory_total_gb']:.1f} GB")
        if gpu_info["vendor"] == "amd":
            print(f"[GPU] HIP Version: {gpu_info.get('hip_version', 'unknown')}")
        else:
            print(f"[GPU] Compute Capability: {gpu_info.get('compute_capability', 'unknown')}")
    
    return True


REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def get_error_name(e: Exception) -> str:

    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def fetch_kernel_from_database(
    run_name: str, problem_id: int, sample_id: int, server_url: str
):
    """
    Intenral to us with our django database
    Return a dict with kernel hash, kernel code, problem_id
    """
    response = requests.get(
        f"{server_url}/get_kernel_by_run_problem_sample/{run_name}/{problem_id}/{sample_id}",
        json={"run_name": run_name, "problem_id": problem_id, "sample_id": sample_id},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert str(response_json["problem_id"]) == str(problem_id)
    return response_json


def fetch_ref_arch_from_problem_id(problem_id, problems, with_name=False) -> str:
    """
    Fetches the reference architecture in string for a given problem_id
    """
    if isinstance(problem_id, str):
        problem_id = int(problem_id)

    problem_path = problems[problem_id]

    # problem_path = os.path.join(REPO_ROOT_PATH, problem)
    if not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file at {problem_path} does not exist.")

    ref_arch = utils.read_file(problem_path)
    if not with_name:
        return ref_arch
    else:
        return (problem_path, ref_arch)


def fetch_ref_arch_from_level_problem_id(level, problem_id, with_name=False):
    PROBLEM_DIR = os.path.join(KERNEL_BENCH_PATH, "level" + str(level))
    dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)
    return fetch_ref_arch_from_problem_id(problem_id, dataset, with_name)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    Works with both NVIDIA CUDA and AMD ROCm GPUs.
    """
    torch.manual_seed(seed)
    # NOTE: this sets on current GPU device (CUDA or ROCm via HIP)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # Set deterministic behavior
    "NOTE: this is not supported for ROCm"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """

    compiled: bool = False
    correctness: bool = False
    metadata: dict = {}
    runtime: float = -1.0  # in ms, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance
    ref_runtime: float = -1.0  # in ms, reference model runtime
    ref_runtime_stats: dict = {}  # reference model runtime statistics
    speedup: float = -1.0  # speedup ratio: ref_runtime / custom_runtime


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """
    fake_filename = f"<original_model_{id(model_original_src)}>"
    
    try:
        lines = model_original_src.splitlines(keepends=True)
        linecache.cache[fake_filename] = (
            len(model_original_src),
            None,
            lines,
            fake_filename,
        )
        
        code_obj = compile(model_original_src, fake_filename, "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        if fake_filename in linecache.cache:
            del linecache.cache[fake_filename]
        return None

    try:
        exec(code_obj, context)
    except Exception as e:
        print(f"Error in executing original code {e}")
        if fake_filename in linecache.cache:
            del linecache.cache[fake_filename]
        return None

    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model_with_tempfile(model_custom_src, entry_point="ModelNew"):
    """
    Writes the provided Python code string to a temporary .py file,
    dynamically imports the module so we can access the modified model class.

    Returns both a Model class and the temporary file. The temporary file must be
    deleted manually be the caller.

    This is needed for:
    - Triton code: compile/exec do not play well with @triton.jit decorator
    - Helion code: requires source file to exist for inspect.getsource() at runtime
    
    Works with both NVIDIA CUDA and AMD ROCm GPUs.
    """

    # Create a temporary named file with a .py extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        # Write the code string into the file
        tmp_file.write(model_custom_src)
        # Capture the path to the file
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    # Create a module specification pointing to our temp file
    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    # Create a new module based on that spec
    temp_module = importlib.util.module_from_spec(spec)
    # Register module in sys.modules (needed for Helion's inspect.getsource())
    sys.modules["temp_module"] = temp_module
    # Execute the code in the module's namespace
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    # Return the object (class, function, etc.) that was defined in the code
    return ModelNew, temp_file


# def load_tilelang_model(
#     model_custom_src: str,
#     context: dict,
#     build_directory: str | None = None
# ):
#     """
#     Load TileLang model using linecache instead of tempfile.
#     This registers the source code in memory so inspect.getsource() works,
#     which is needed for TileLang's JIT decorator.
#     """
#     if build_directory:
#         model_custom_src = (
#             "import os\n"
#             f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
#             + model_custom_src
#         )
#
#     # Register source so inspect.getsource works
#     fake_fname = (
#         f"/tmp/tilelang_kernel_"
#         f"{hashlib.md5(model_custom_src.encode()).hexdigest()}.py"
#     )
#     # linecache expects a list with trailing newlines
#     linecache.cache[fake_fname] = (
#         len(model_custom_src),
#         None,
#         model_custom_src.splitlines(True),
#         fake_fname,
#     )
#
#     code_obj = compile(model_custom_src, fake_fname, "exec")
#     exec(code_obj, context)
#     return context["ModelNew"]


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    fake_filename = f"<generated_model_{id(model_custom_src)}>"
    
    try:
        lines = model_custom_src.splitlines(keepends=True)
        linecache.cache[fake_filename] = (
            len(model_custom_src),
            None,
            lines,
            fake_filename,
        )
        
        code_obj = compile(model_custom_src, fake_filename, "exec")
        exec(code_obj, context)
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        if fake_filename in linecache.cache:
            del linecache.cache[fake_filename]
        return None
    except Exception as e:
        print(f"Error executing custom model: {e}")
        if fake_filename in linecache.cache:
            del linecache.cache[fake_filename]
        raise

    ModelNew = context.get("ModelNew")
    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    """
    Clean up environment, GPU cache, and compiled extensions after evaluation.
    Works with both NVIDIA CUDA and AMD ROCm GPUs.
    """
    # Clean up linecache entries
    fake_filenames = [k for k in linecache.cache.keys() if k.startswith(("<generated_model_", "<original_model_"))]
    for fname in fake_filenames:
        del linecache.cache[fname]
    
    # Clean up temp_module from sys.modules if it exists (used by Helion)
    if "temp_module" in sys.modules:
        del sys.modules["temp_module"]
    
    del curr_context
    
    # Clean up GPU memory (works for both CUDA and ROCm)
    if torch.cuda.is_available():
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device=device)
                torch.cuda.synchronize(device=device)
        except Exception:
            # Ignore cleanup errors
            pass
    
    # Clean up temporary file
    if tempfile:
        try:
            tempfile.close()
            if os.path.exists(tempfile.name):
                os.remove(tempfile.name)
        except Exception:
            pass


def build_compile_cache_legacy(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible

    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
    custom_model_src: str, verbose: bool = False, build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(
        ["python", tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)

    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode("utf-8"))
        print("[CPU Precompile] stderr: \n", stderr.decode("utf-8"))

    return returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def _process_input_tensor(tensor, device, backend):
    """
    Helper function to move tensors to the correct device and apply backend-specific dtype casting.
    
    Args:
        tensor: Input tensor or non-tensor value
        device: Target GPU device (CUDA or ROCm)
        backend: Backend type (e.g., 'cuda', 'triton', 'cute', 'helion')
    
    Returns:
        Processed tensor on correct device with correct dtype, or original value if not a tensor
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    
    # Preserve integer dtypes for labels/targets (e.g., classification losses)
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        return tensor.to(device=device)
    
    # Backend-specific dtype handling
    backend_lower = backend.lower() if backend else "cuda"
    
    # Helion may benefit from fp16 on some AMD GPUs
    # if backend_lower == "helion":
    #     return tensor.to(device=device, dtype=torch.float16)
    
    # Default for all other backends and float types
    return tensor.to(device=device)


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = False,
    measure_performance: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),  # have to run on GPU (CUDA or ROCm)
    backend: str = "cuda",  # can be 'cuda', 'triton', 'cute', or 'helion'
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model.
    
    Supports both NVIDIA CUDA and AMD ROCm GPUs.

    Args:
        original_model_src: Source code of the reference PyTorch model
        custom_model_src: Source code of the optimized model with custom kernels
        seed_num: Random seed for reproducibility
        num_correct_trials: Number of trials for correctness check
        num_perf_trials: Number of trials for performance measurement
        verbose: Enable verbose logging
        measure_performance: Whether to measure and compare performance
        build_dir: Directory for caching compiled kernels
        device: GPU device to run evaluation on (CUDA or ROCm)
        backend: One of 'cuda', 'triton', 'cute', or 'helion'
    
    Returns:
        KernelExecResult with compilation status, correctness, and performance metrics
    """
    # Check GPU availability (works for both CUDA and ROCm)
    if not check_gpu_available(verbose=verbose):
        raise RuntimeError("No GPU available (CUDA or ROCm), cannot run Eval")
    
    # Get GPU vendor info for metadata
    gpu_vendor = get_gpu_vendor()
    gpu_info = get_gpu_info(device if isinstance(device, int) else None)
    
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # Set GPU device (works for both CUDA and ROCm via HIP)
    torch.cuda.set_device(device)
    
    # Backends that use tempfile approach
    # - triton: @triton.jit decorator requires file-based import
    # - cute: CUTLASS requires file-based compilation
    # - helion: @helion.kernel decorator requires inspect.getsource()
    backend_lower = backend.lower()
    uses_tempfile = backend_lower in ["triton", "cute", "helion"]
    
    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
    metadata["gpu_vendor"] = gpu_vendor
    metadata["backend"] = backend_lower
    
    # Add vendor-specific info
    # AMD ROCm specific info
    if gpu_vendor == "amd":
        metadata["hip_version"] = gpu_info.get("hip_version", "unknown")
        metadata["gcn_arch"] = gpu_info.get("gcn_arch", "unknown")
    else:
        metadata["compute_capability"] = gpu_info.get("compute_capability", "unknown")

    if uses_tempfile:
        # Set device visibility for triton/cute/helion
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert (
                device.type == "cuda"
            ), "GPU is not available on device, cannot run Eval"
            device_num = device.index if device.index is not None else 0
        else:
            raise ValueError(
                f"device must be an int or torch.device, got {type(device)}"
            )
        
        # Set device visibility
        # For ROCm, use HIP_VISIBLE_DEVICES; for CUDA, use CUDA_VISIBLE_DEVICES
        if gpu_vendor == "amd":
            os.environ["HIP_VISIBLE_DEVICES"] = str(device_num)
            os.environ["ROCR_VISIBLE_DEVICES"] = str(device_num)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    
    context = {}
    "Context is a dictionary that stores the context of the evaluation"
    if verbose:
        vendor_str = "AMD ROCm" if gpu_vendor == "amd" else "NVIDIA CUDA"
        print(f"[Eval] Start Evaluation on device: {device} ({vendor_str})")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    
    # Convert inputs to appropriate dtypes for GPU computation
    init_inputs = [_process_input_tensor(x, device, backend) for x in init_inputs]
    
    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    
    if verbose:
        backend_name = backend_lower.upper()
        print(f"[Eval] Loading and Compiling New Model with Custom {backend_name} Kernel")

    # This is where compilation happens
    try:
        # Enable device-side assertions for debugging
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        tempfile = None
        
        if backend_lower in ["triton", "cute", "helion"]:
            # Use tempfile approach for:
            # - triton: @triton.jit decorator requires file-based import
            # - cute: CUTLASS requires file-based compilation  
            # - helion: @helion.kernel decorator requires inspect.getsource()
            ModelNew, tempfile = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            # Default CUDA backend (inline compilation)
            ModelNew = load_custom_model(custom_model_src, context, build_dir)
        
        torch.cuda.synchronize(device=device)
    except Exception as e:
        print(
            f"Failed to compile custom {backend_lower.upper()} kernel: Record as compilation failure. \nError: {e}"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # This is a lock file error, likely due to concurrent compilation
            # This does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device, tempfile)
            return None
        else:
            metadata["compilation_error_name"] = get_error_name(e)
            metadata["compilation_error"] = str(e)
            graceful_eval_cleanup(context, device, tempfile)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # At this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            
            # Move models to GPU
            original_model = original_model.to(device=device)
            custom_model = custom_model.to(device=device)
            torch.cuda.synchronize(device=device)
        
        if verbose:
            print(f"[Eval] New Model with Custom {backend_lower.upper()} Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom {backend_lower.upper()} kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context, device, tempfile)
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            backend=backend,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                
                # Measure reference model performance
                if verbose:
                    print("[Eval] Measuring Reference Model Performance")
                set_seed(seed_num)
                ref_inputs = get_inputs()
                ref_inputs = [_process_input_tensor(x, device, backend) for x in ref_inputs]
                ref_model = original_model.to(device=device)
                torch.cuda.synchronize(device=device)
                
                ref_elapsed_times = time_execution_with_gpu_event(
                    ref_model,
                    *ref_inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                ref_runtime_stats = get_timing_stats(ref_elapsed_times, device=device)
                
                if verbose:
                    print(f"[Eval] Reference Performance Stats: {ref_runtime_stats}")
                kernel_exec_result.ref_runtime = ref_runtime_stats["mean"]
                kernel_exec_result.ref_runtime_stats = ref_runtime_stats
                
                # Measure custom model performance
                if verbose:
                    print("[Eval] Measuring Custom Model Performance")
                set_seed(seed_num)
                inputs = get_inputs()
                inputs = [_process_input_tensor(x, device, backend) for x in inputs]
                
                model_new = custom_model.to(device=device)
                torch.cuda.synchronize(device=device)

                elapsed_times = time_execution_with_gpu_event(
                    model_new,
                    *inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Custom Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats
                
                # Calculate speedup
                if kernel_exec_result.runtime > 0 and kernel_exec_result.ref_runtime > 0:
                    kernel_exec_result.speedup = kernel_exec_result.ref_runtime / kernel_exec_result.runtime
                    if verbose:
                        print(f"[Eval] Speedup: {kernel_exec_result.speedup:.2f}x")
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = str(e)

    graceful_eval_cleanup(context, device, tempfile)
    return kernel_exec_result


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def time_execution_with_gpu_event(
    kernel_fn: callable,
    *args,
    num_warmup: int = 3,
    num_trials: int = 10,
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a GPU kernel function over multiple trials using torch.cuda.Event.
    
    Works with both NVIDIA CUDA and AMD ROCm GPUs.
    The torch.cuda.Event API is supported by both backends.

    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_warmup: Number of warmup iterations
        num_trials: Number of timing trials to run
        verbose: Whether to print per-trial timing info
        device: GPU device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Normalize device to an index for torch.cuda APIs
    if isinstance(device, torch.device):
        cuda_device = device.index if device.type == "cuda" else None
    else:
        cuda_device = device

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=cuda_device)

    gpu_vendor = get_gpu_vendor()
    vendor_str = "ROCm" if gpu_vendor == "amd" else "CUDA"
    device_name = "unknown"
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(cuda_device)
        except Exception:
            device_name = "unknown"
    print(
        f"[Profiling] Using device: {device} {device_name} ({vendor_str}), "
        f"warmup {num_warmup}, trials {num_trials}"
    )
    elapsed_times = []

    # Actual trials
    use_events = torch.cuda.is_available()
    if use_events:
        try:
            _ = torch.cuda.Event(enable_timing=True)
        except Exception:
            use_events = False

    if use_events:
        for trial in range(num_trials):
            # Create event markers
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            kernel_fn(*args)
            end_event.record()

            # Synchronize to ensure the events have completed
            torch.cuda.synchronize(device=cuda_device)

            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            if verbose:
                print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)
    else:
        # ROCm fallback if CUDA events are not supported
        for trial in range(num_trials):
            start_time = time.perf_counter()
            kernel_fn(*args)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=cuda_device)
            elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
            if verbose:
                print(f"Trial {trial + 1}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)

    return elapsed_times


# Alias for backward compatibility
time_execution_with_cuda_event = time_execution_with_gpu_event


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose=False,
    seed=42,
    device=None,
    backend="cuda",
) -> KernelExecResult:
    """
    Run the model and check correctness against reference implementation.
    
    Assumes models are already loaded and compiled (done in the caller).
    Works with both NVIDIA CUDA and AMD ROCm GPUs.

    Args:
        original_model_instance: Reference PyTorch model instance
        new_model_instance: Custom kernel model instance to validate
        get_inputs_fn: Function that returns input tensors
        metadata: Dict to store evaluation metadata
        num_correct_trials: Number of trials with different random inputs
        verbose: Enable verbose logging
        seed: Random seed for reproducibility
        device: GPU device (CUDA or ROCm)
        backend: Backend type ('cuda', 'triton', 'cute', 'helion')
    
    Returns:
        KernelExecResult with correctness status and metadata
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")
            
            # if backend.lower() == "tilelang":
            #     torch.set_default_dtype(torch.float16)

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            # Convert inputs to appropriate dtypes for GPU computation
            inputs = [_process_input_tensor(x, device, backend) for x in inputs]

            set_seed(trial_seed)
            # if backend.lower() == "tilelang":
            #     try:
            #         model = original_model_instance.to(device=device, dtype=torch.float16)
            #     except Exception as e:
            #         # TileLang JIT kernels may not support .to(), already on GPU
            #         if verbose:
            #             print(f"[Info] Line 771 - Could not call .to() on original model (TileLang): {e}")
            #             print("[Traceback] From run_and_check_correctness - line 771:")
            #             traceback.print_exc()
            #         model = original_model_instance
            # else:
            model = original_model_instance.to(device=device)

            set_seed(trial_seed)
            # if backend.lower() == "tilelang":
            #     try:
            #         model_new = new_model_instance.to(device=device, dtype=torch.float16)
            #     except Exception as e:
            #         # TileLang JIT kernels may not support .to(), already on GPU
            #         if verbose:
            #             print(f"[Info] Line 777 - Could not call .to() on custom model (TileLang): {e}")
            #             print("[Traceback] From run_and_check_correctness - line 777:")
            #             traceback.print_exc()
            #         model_new = new_model_instance
            # else:
            model_new = new_model_instance.to(device=device)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=1e-02, rtol=1e-02
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")
                print("\n[Full Traceback]:")
                traceback.print_exc()
                print("\n")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                metadata["runtime_error_name"] = get_error_name(e)
                # Also store the full traceback in metadata for debugging
                metadata["runtime_error_traceback"] = traceback.format_exc()
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata


def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


################################################################################
# Performance Eval
################################################################################


def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: list[str], baseline_time_filepath: str
) -> dict:
    """
    Fetch the baseline time from the time
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem_name = dataset[problem_id].split("/")[-1]
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Works with both NVIDIA CUDA and AMD ROCm GPUs.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: GPU device (CUDA or ROCm), record device info
        
    Returns:
        Dict containing mean, std, min, max, num_trials, and device info
        All timing values are in milliseconds.
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device is not None and torch.cuda.is_available():
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)
        stats["gpu_vendor"] = get_gpu_vendor()

    return stats


# if __name__ == "__main__":
# fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")
# print(fetch_ref_arch_from_level_problem_id("2", 1, with_name=True))
# fetch_baseline_time("level1", 0, ["1_Square_matrix_multiplication_.py"], "tests/baseline_time_matx3.json")