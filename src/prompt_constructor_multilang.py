import os
from .utils import read_file
from .prompts.amd_gpu.prompts import (
    HIGH_CORRECT_PROMPT,
    RDNA4_PROMPT,
    QUANT_OP_PROMPT,
)

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
    ref_arch_src: str, gpu_name: str, vendor: str = "nvidia"
) -> str:
    """
    Similar to prompt_generate_custom_triton_from_prompt_template,
    but with hardware information for the given GPU
    """
    if vendor.lower() == "amd":
        return _build_triton_prompt_with_amd_hw(ref_arch_src, gpu_name)

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


def _build_triton_prompt_with_amd_hw(ref_arch_src: str, gpu_name: str) -> str:
    """
    Build the Triton prompt with AMD hardware info and AMD-specific guidance.
    """
    arch = ref_arch_src

    example_arch_path = os.path.join(REPO_TOP_PATH, f"src/prompts/model_ex_add.py")
    example_new_arch_path = os.path.join(
        REPO_TOP_PATH, f"src/prompts/model_new_ex_add_triton.py"
    )

    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    prompt = TRITON_PROBLEM_STATEMENT

    if example_arch and example_new_arch:
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom Triton kernels in torch: The example given architecture is: \n
        ``` \n
        {example_arch}
        ``` \n
        The example new arch with custom Triton kernels looks like this: 
        ```
        {example_new_arch}
        ``` \n
        """

    # Load AMD GPU spec info
    amd_spec_file_path = os.path.join(
        REPO_TOP_PATH, "src/prompts/hardware/amd_gpu_specs.py"
    )
    amd_spec_info_src = read_file(amd_spec_file_path)
    amd_spec_local = {}
    exec(amd_spec_info_src, {}, amd_spec_local)
    amd_gpu_spec_info = amd_spec_local.get("AMD_GPU_SPEC_INFO", {})
    amd_gpu_definitions = amd_spec_local.get("AMD_GPU_DEFINITIONS", {})
    amd_gpu_best_practices = amd_spec_local.get("AMD_GPU_BEST_PRACTICES", [])

    prompt += f"""
    Here is some information about the underlying hardware that you should keep in mind. \n\n
The GPU that will run the kernel is AMD {gpu_name}.\n\n"""

    gpu_key = None
    for key in amd_gpu_spec_info.keys():
        if key.lower() in gpu_name.lower():
            gpu_key = key
            break
    if gpu_key is None and amd_gpu_spec_info:
        gpu_key = next(iter(amd_gpu_spec_info))

    if gpu_key:
        gpu_info = amd_gpu_spec_info[gpu_key]
        for key, value in gpu_info.items():
            prompt += f"- {key}: {value}\n"

    if amd_gpu_definitions:
        prompt += "\nHere are some concepts about AMD GPU architecture:\n\n"
        for key, value in amd_gpu_definitions.items():
            prompt += f"- {key}: {value}\n"

    if amd_gpu_best_practices:
        prompt += "\nHere are some best practices for writing Triton kernels on AMD GPUs:\n\n"
        for best_practice in amd_gpu_best_practices:
            prompt += f"- {best_practice}\n"

    # Add AMD-specific guidance
    prompt += "\n\n### AMD GPU Optimization Guidance\n\n"
    prompt += HIGH_CORRECT_PROMPT

    gpu_name_lower = gpu_name.lower()
    if "gfx12" in gpu_name_lower or "rdna4" in gpu_name_lower or "rx 9" in gpu_name_lower or "rx9" in gpu_name_lower:
        prompt += "\n\n" + RDNA4_PROMPT
    if "mi300" in gpu_name_lower or "mi355" in gpu_name_lower or "mi3" in gpu_name_lower:
        prompt += "\n\n" + HIGH_CORRECT_PROMPT

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arch}
    ```
    """
    prompt += TRITON_PROBLEM_INSTRUCTION
    return prompt


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

def get_prompt_for_backend(
    ref_arch_src: str,
    backend: str = "triton",
    gpu_name: str | None = None,
    vendor: str = "nvidia",
) -> str:
    """
    Unified API to get prompt for any supported backend
    
    Args:
        ref_arch_src: Reference architecture source code
        backend: One of 'triton', 'cute'  (tilelang removed - not working)
    
    Returns:
        Prompt string for the specified backend
    """
    backend_lower = backend.lower()
    
    if backend_lower == "triton":
        if gpu_name:
            return prompt_generate_prompt_with_hardware_info_from_template_triton(
                ref_arch_src, gpu_name, vendor=vendor
            )
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

    ref_arch_src = read_file(os.path.join(KERNEL_BENCH_PATH, f"level1/19_ReLU.py"))
    assert len(ref_arch_src) > 0, "ref_arch_src is empty"
    
    prompt = get_prompt_for_backend(ref_arch_src, backend)
    print(f"\n{'='*80}\n{backend.upper()} PROMPT:\n{'='*80}\n")
    print(prompt)
    
    # Write prompt to temp file
    temp_file_path = os.path.join(REPO_TOP_PATH, "scratch", f"prompt_{backend}_draft.txt")
    os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
    with open(temp_file_path, "w") as f:
        f.write(prompt)
    print(f"\nPrompt written to: {temp_file_path}")


if __name__ == "__main__":
    main()



