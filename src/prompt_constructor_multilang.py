import os
from .utils import read_file

"""
Multi-Language Prompt Constructor

Supports: Triton, CuTe (TileLang currently disabled/commented out)

Design principles: 
- To evaluate base model performance on KernelBench, we use the simplest prompt possible to guide model output to generated desired output format.
- However, we do not do extensive prompt engineering or few-shot examples in the LLM to steer behaviour. 
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def get_arch_definition_from_file(arch_path):
    arch_src = read_file(arch_path)
    return get_arch_definition(arch_src)


def get_arch_definition(arch_src):
    """
    Construct torch definition from original torch nn.Module definition
    """
    prompt = f"Here is a pytorch defintion of a neural network architecture in the file model.py: ```{arch_src}```\n"
    return prompt


################################################################################
# Triton Backend
################################################################################

TRITON_PROBLEM_STATEMENT = """You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

TRITON_PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

TRITON_PROBLEM_STATEMENT_CLEANED = """You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

TRITON_PROBLEM_INSTRUCTION_CLEANED = """
Optimize the architecture named Model with custom Triton kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

################################################################################
# Triton Optimization (for level7 - when input already contains Triton kernels)
################################################################################

TRITON_OPTIMIZE_PROBLEM_STATEMENT = """You are given an architecture that already uses Triton kernels. Your task is to optimize the existing Triton kernels for better performance. \n
    You should analyze the current Triton implementation and apply optimization techniques such as:
    - Using @triton.autotune decorator to automatically tune block sizes and other parameters
    - Optimizing memory access patterns (coalesced access, reduce bank conflicts)
    - Adjusting block sizes, num_warps, and num_stages parameters
    - Exploiting data reuse through shared memory or register tiling
    - Minimizing memory transfers between global and shared memory
    - Using block-level primitives for reductions
    - Fusing additional operations into existing kernels if applicable
    - Algorithmic improvements (such as online softmax, flash attention patterns)
    
You must maintain functional correctness while improving performance.\n
"""

TRITON_OPTIMIZE_PROBLEM_INSTRUCTION = """
Optimize the Triton kernels in the architecture named Model for better performance! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. The output must be functionally equivalent to the input (produce the same results). Just output the new model code, no other text, and NO testing code! \n
"""


def is_triton_implementation(src: str) -> bool:
    """
    Check if the source code already contains Triton kernel implementations.
    This is used to distinguish between PyTorch-only code (level1-6) and 
    Triton-based code (level7).
    
    Args:
        src: Source code string to check
        
    Returns:
        True if the code contains Triton kernels, False otherwise
    """
    # Check for common Triton kernel indicators
    triton_indicators = [
        "@triton.jit",
        "triton.jit",
        "tl.load",
        "tl.store",
        "tl.program_id",
        "triton.language",
    ]
    return any(indicator in src for indicator in triton_indicators)


def prompt_generate_custom_triton(
    arc_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = TRITON_PROBLEM_STATEMENT

    assert (
        "@triton.jit" in example_new_arch_src
    ), "Example new arch must contain Triton kernel"

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom Triton kernels looks like this: \n
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += TRITON_PROBLEM_INSTRUCTION
    return prompt


def prompt_generate_custom_triton_fewshot_and_template(
    ref_arch_src: str, shots: list
) -> str:
    raise NotImplementedError("This function has not been implemented yet")


def prompt_generate_ex_with_CoT_template_triton(ref_arch_src: str, cot_example: str) -> str:
    raise NotImplementedError("This function has not been implemented yet")


def prompt_generate_custom_triton_from_prompt_template(ref_arch_src: str) -> str:
    """
    Using prompt example (an element-wise addition) for prompt templates
    The most basic form of example just to show LLM the task and the expected output format
    """
    arch = ref_arch_src

    # path to prompt template, show an example of Model (torch specifications) and ModelNew (torch + custom Triton kernels)
    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_add.py")
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add_triton.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        raise FileNotFoundError(
            f"Example new architecture file not found: {example_new_arch_path}"
        )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_triton(arch, example_arch, example_new_arch)


def prompt_generate_prompt_with_hardware_info_from_template_triton(
    ref_arch_src: str, gpu_name: str
) -> str:
    """
    Similar to prompt_generate_custom_triton_from_prompt_template,
    but with hardware information for the given GPU
    """
    arch = ref_arch_src

    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_add.py")
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add_triton.py"
    )
    gpu_spec_file_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/hardware/gpu_specs.py"
    )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)
    gpu_spec_info = read_file(gpu_spec_file_path)

    return prompt_generate_prompt_with_hardware_info_triton(
        ref_arch_src=arch,
        gpu_name=gpu_name,
        example_arch_src=example_arch,
        example_new_arch_src=example_new_arch,
        gpu_spec_info_src=gpu_spec_info,
    )


def prompt_generate_prompt_with_hardware_info_triton(
    ref_arch_src: str,
    gpu_name: str,
    example_arch_src: str,
    example_new_arch_src: str,
    gpu_spec_info_src: str,
) -> str:
    """
    Generate a prompt with hardware information for the given GPU
    gpu_spec_info_src: str of the gpu spec src file
    """
    local_dict = {}
    exec(gpu_spec_info_src, {}, local_dict)

    GPU_SPEC_INFO = local_dict.get("GPU_SPEC_INFO")
    GPU_DEFINITIONS = local_dict.get("GPU_DEFINITIONS")
    GPU_BEST_PRACTICES = local_dict.get("GPU_BEST_PRACTICES")

    if not GPU_SPEC_INFO or not GPU_DEFINITIONS or not GPU_BEST_PRACTICES:
        raise ValueError(
            "GPU_SPEC_INFO or GPU_DEFINITIONS or GPU_BEST_PRACTICES not found in gpu_spec_info_src"
        )

    assert gpu_name in GPU_SPEC_INFO, f"GPU name {gpu_name} not found in GPU_SPEC_INFO"

    prompt = TRITON_PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom Triton kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    curr_gpu_spec_info = GPU_SPEC_INFO[gpu_name]
    gpu_architecture = curr_gpu_spec_info.get("GPU Architecture")
    prompt += f"""
    Here is some information about the underlying hardware that you should keep in mind. \n\n
The GPU that will run the kernel is NVIDIA {gpu_name}, {gpu_architecture} architecture.\n\n"""

    for key, value in curr_gpu_spec_info.items():
        if key == "GPU Architecture":
            continue
        prompt += f"""- We have {value} of {key}.\n"""

    prompt += f"""\n\n
Here are some concepts about the GPU architecture that could be helpful: \n\n"""
    for key, value in GPU_DEFINITIONS.items():
        prompt += f"""- {key}: {value}\n"""

    prompt += f"""\n\n
Here are some best practices for writing Triton kernels on GPU: \n\n"""
    for best_practice in GPU_BEST_PRACTICES:
        prompt += f"""- {best_practice}\n"""

    prompt += f"""
    You are given the following architecture: \n
    ```
    {ref_arch_src}
    ```
    """

    prompt += TRITON_PROBLEM_INSTRUCTION
    return prompt


def prompt_fix_compile_triton(ref_arch_src, custom_kernel, metadata):
    prompt = TRITON_PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed to compile:
    ```
    {custom_kernel}
    ```
    Here's the metadata of the compilation error:
    ```
    {metadata}
    ```
    
    Please fix the compilation error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt


def prompt_fix_correctness_triton(ref_arch_src, custom_kernel, metadata):
    prompt = TRITON_PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed correctness:
    ```
    {custom_kernel}
    ```
    Here's the metadata of the correctness error:
    ```
    {metadata}
    ```
    Please consider how your custom Triton kernels are implemented, how it is different from the reference implementation, and fix the correctness error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt


def prompt_generate_optimized_triton(ref_arch_src: str) -> str:
    """
    Generate prompt for optimizing existing Triton kernels (level7).
    This is used when the input already contains Triton kernel implementations.
    
    Args:
        ref_arch_src: Reference architecture source code containing Triton kernels
        
    Returns:
        Prompt string for optimizing the Triton kernels
    """
    prompt = TRITON_OPTIMIZE_PROBLEM_STATEMENT
    
    prompt += f"""
    You are given the following architecture with Triton kernels: \n
    ```
    {ref_arch_src}
    ```
    """
    prompt += TRITON_OPTIMIZE_PROBLEM_INSTRUCTION
    return prompt


def prompt_generate_optimized_triton_from_prompt_template(ref_arch_src: str) -> str:
    """
    Generate prompt for optimizing existing Triton kernels with an example (level7).
    Shows an example of optimizing a basic Triton kernel to an autotuned version.
    
    Args:
        ref_arch_src: Reference architecture source code containing Triton kernels
        
    Returns:
        Prompt string for optimizing the Triton kernels
    """
    prompt = TRITON_OPTIMIZE_PROBLEM_STATEMENT
    
    # Add an inline example of Triton optimization
    prompt += """
    Here's an example of optimizing a Triton kernel with autotuning:
    
    Before optimization:
    ```python
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    ```
    
    After optimization (with autotuning):
    ```python
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
            triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
            triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        block_start = tl.program_id(0) * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)
    ```
    """
    
    prompt += f"""
    You are given the following architecture with Triton kernels to optimize: \n
    ```
    {ref_arch_src}
    ```
    """
    prompt += TRITON_OPTIMIZE_PROBLEM_INSTRUCTION
    return prompt


################################################################################
# TileLang Backend - COMMENTED OUT (not working currently)
################################################################################

# TILELANG_PROBLEM_STATEMENT = """You write custom TileLang kernels to replace the pytorch operators in the given architecture to get speedups. \n
#     You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom TileLang kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
# """
# 
# TILELANG_PROBLEM_INSTRUCTION = """
# Optimize the architecture named Model with custom TileLang kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
# """
# 
# TILELANG_PROBLEM_STATEMENT_CLEANED = """You write custom TileLang kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom TileLang kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
# """
# 
# TILELANG_PROBLEM_INSTRUCTION_CLEANED = """
# Optimize the architecture named Model with custom TileLang kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
# """
# 
# 
# def prompt_generate_custom_tilelang(
#     arc_src: str, example_arch_src: str, example_new_arch_src: str
# ) -> str:
#     prompt = TILELANG_PROBLEM_STATEMENT
# 
#     if example_arch_src != "" and example_new_arch_src != "":
#         prompt += f"""
#         Here's an example to show you the syntax of inline embedding custom TileLang kernels in torch: The example given architecture is: \n
#         ``` \n
#         {example_arch_src}
#         ``` \n
#         The example new arch with custom TileLang kernels looks like this: \n
#         ```
#         {example_new_arch_src}
#         ``` \n
#         """
# 
#     prompt += f"""
#     You are given the following architecture: \n
#     ```
#     {arc_src}
#     ```
#     """
#     prompt += TILELANG_PROBLEM_INSTRUCTION
#     return prompt
# 
# 
# def prompt_generate_custom_tilelang_from_prompt_template(ref_arch_src: str) -> str:
#     """
#     Using prompt example for TileLang
#     Note: You'll need to create a TileLang example file similar to the Triton one
#     """
#     arch = ref_arch_src
# 
#     # TODO: Create model_new_ex_add_tilelang.py example file
#     example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_add.py")
#     example_new_arch_path = os.path.join(
#         REPO_TOP_PATH, f"src/prompts/model_new_ex_add_tilelang.py"
#     )
# 
#     if not os.path.exists(example_arch_path):
#         raise FileNotFoundError(
#             f"Example architecture file not found: {example_arch_path}"
#         )
#     if not os.path.exists(example_new_arch_path):
#         # For now, use a basic template without examples if file doesn't exist
#         return prompt_generate_custom_tilelang(arch, "", "")
# 
#     example_arch = read_file(example_arch_path)
#     example_new_arch = read_file(example_new_arch_path)
# 
#     return prompt_generate_custom_tilelang(arch, example_arch, example_new_arch)
# 
# 
# def prompt_fix_compile_tilelang(ref_arch_src, custom_kernel, metadata):
#     prompt = TILELANG_PROBLEM_STATEMENT
#     prompt += f"""
#     With the following architecture:
#     ```
#     {ref_arch_src}
#     ```
#     You generated the following solution and it failed to compile:
#     ```
#     {custom_kernel}
#     ```
#     Here's the metadata of the compilation error:
#     ```
#     {metadata}
#     ```
#     
#     Please fix the compilation error in the new model code. Please output the corrected code in codeblocks.
#     """
#     return prompt
# 
# 
# def prompt_fix_correctness_tilelang(ref_arch_src, custom_kernel, metadata):
#     prompt = TILELANG_PROBLEM_STATEMENT
#     prompt += f"""
#     With the following architecture:
#     ```
#     {ref_arch_src}
#     ```
#     You generated the following solution and it failed correctness:
#     ```
#     {custom_kernel}
#     ```
#     Here's the metadata of the correctness error:
#     ```
#     {metadata}
#     ```
#     Please consider how your custom TileLang kernels are implemented, how it is different from the reference implementation, and fix the correctness error in the new model code. Please output the corrected code in codeblocks.
#     """
#     return prompt


################################################################################
# CuTe Backend
################################################################################

CUTE_PROBLEM_STATEMENT = """You write custom CuTe (CUTLASS) kernels to replace the pytorch operators in the given architecture to get speedups. \n
    You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CuTe kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

CUTE_PROBLEM_INSTRUCTION = """
Optimize the architecture named Model with custom CuTe (CUTLASS) kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

CUTE_PROBLEM_STATEMENT_CLEANED = """You write custom CuTe (CUTLASS) kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CuTe kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

CUTE_PROBLEM_INSTRUCTION_CLEANED = """
Optimize the architecture named Model with custom CuTe (CUTLASS) kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""


def prompt_generate_custom_cute(
    arc_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = CUTE_PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CuTe (CUTLASS) kernels in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CuTe kernels looks like this: \n
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arc_src}
    ```
    """
    prompt += CUTE_PROBLEM_INSTRUCTION
    return prompt


def prompt_generate_custom_cute_from_prompt_template(ref_arch_src: str) -> str:
    """
    Using prompt example for CuTe
    Note: You'll need to create a CuTe example file
    """
    arch = ref_arch_src

    # TODO: Create model_new_ex_add_cute.py example file
    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_add.py")
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add_cute.py"
    )

    if not os.path.exists(example_arch_path):
        raise FileNotFoundError(
            f"Example architecture file not found: {example_arch_path}"
        )
    if not os.path.exists(example_new_arch_path):
        # For now, use a basic template without examples if file doesn't exist
        return prompt_generate_custom_cute(arch, "", "")

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_custom_cute(arch, example_arch, example_new_arch)


def prompt_fix_compile_cute(ref_arch_src, custom_kernel, metadata):
    prompt = CUTE_PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed to compile:
    ```
    {custom_kernel}
    ```
    Here's the metadata of the compilation error:
    ```
    {metadata}
    ```
    
    Please fix the compilation error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt


def prompt_fix_correctness_cute(ref_arch_src, custom_kernel, metadata):
    prompt = CUTE_PROBLEM_STATEMENT
    prompt += f"""
    With the following architecture:
    ```
    {ref_arch_src}
    ```
    You generated the following solution and it failed correctness:
    ```
    {custom_kernel}
    ```
    Here's the metadata of the correctness error:
    ```
    {metadata}
    ```
    Please consider how your custom CuTe kernels are implemented, how it is different from the reference implementation, and fix the correctness error in the new model code. Please output the corrected code in codeblocks.
    """
    return prompt


################################################################################
# Unified API
################################################################################

def get_prompt_for_backend(ref_arch_src: str, backend: str = "triton") -> str:
    """
    Unified API to get prompt for any supported backend
    
    This function automatically detects whether the input is:
    - PyTorch-only code (level1-6): Uses prompts to convert PyTorch to custom kernels
    - Triton-based code (level7): Uses prompts to optimize existing Triton kernels
    
    Args:
        ref_arch_src: Reference architecture source code
        backend: One of 'triton', 'cute'  (tilelang removed - not working)
    
    Returns:
        Prompt string for the specified backend
    """
    backend_lower = backend.lower()
    
    if backend_lower == "triton":
        # Check if the input already contains Triton kernels (level7)
        if is_triton_implementation(ref_arch_src):
            # Input is already Triton-based, use optimization prompt
            return prompt_generate_optimized_triton_from_prompt_template(ref_arch_src)
        else:
            # Input is PyTorch-only, use conversion prompt
            return prompt_generate_custom_triton_from_prompt_template(ref_arch_src)
    # elif backend_lower == "tilelang":
    #     return prompt_generate_custom_tilelang_from_prompt_template(ref_arch_src)
    elif backend_lower == "cute":
        return prompt_generate_custom_cute_from_prompt_template(ref_arch_src)
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. Must be one of: 'triton', 'cute'"
        )


################################################################################
# Main (for testing)
################################################################################

def main():
    gpu_name = "L40S"
    backend = "triton"  # Change this to test different backends

    # Test with PyTorch-only code (level1-6)
    print(f"\n{'='*80}")
    print("TEST 1: PyTorch-only code (level1-6) - should use conversion prompt")
    print(f"{'='*80}\n")
    
    ref_arch_src = read_file(os.path.join(KERNEL_BENCH_PATH, f"level1/19_ReLU.py"))
    assert len(ref_arch_src) > 0, "ref_arch_src is empty"
    
    is_triton = is_triton_implementation(ref_arch_src)
    print(f"Is Triton implementation: {is_triton}")
    
    prompt = get_prompt_for_backend(ref_arch_src, backend)
    print(f"\n{backend.upper()} PROMPT (first 500 chars):\n")
    print(prompt[:500] + "...")
    
    # Write prompt to temp file
    temp_file_path = os.path.join(REPO_TOP_PATH, "scratch", f"prompt_{backend}_pytorch_draft.txt")
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        f.write(prompt)
    print(f"\nPrompt written to: {temp_file_path}")
    
    # Test with Triton code (level7)
    print(f"\n{'='*80}")
    print("TEST 2: Triton-based code (level7) - should use optimization prompt")
    print(f"{'='*80}\n")
    
    level7_path = os.path.join(KERNEL_BENCH_PATH, f"level7/100_GELU_And_Mul.py")
    if os.path.exists(level7_path):
        ref_arch_src_triton = read_file(level7_path)
        assert len(ref_arch_src_triton) > 0, "ref_arch_src_triton is empty"
        
        is_triton = is_triton_implementation(ref_arch_src_triton)
        print(f"Is Triton implementation: {is_triton}")
        
        prompt_triton = get_prompt_for_backend(ref_arch_src_triton, backend)
        print(f"\n{backend.upper()} OPTIMIZATION PROMPT (first 500 chars):\n")
        print(prompt_triton[:500] + "...")
        
        # Write prompt to temp file
        temp_file_path = os.path.join(REPO_TOP_PATH, "scratch", f"prompt_{backend}_triton_draft.txt")
        with open(temp_file_path, "w") as f:
            f.write(prompt_triton)
        print(f"\nPrompt written to: {temp_file_path}")
    else:
        print(f"Level7 file not found: {level7_path}")


if __name__ == "__main__":
    main()



