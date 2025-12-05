import gradio as gr
import requests
import time
import os
import re
from typing import Optional, Dict, Any, List, Tuple

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

class KernelBenchGradioApp:
    def __init__(self, api_base_url: str = "http://localhost:8009"):
        self.api_base_url = api_base_url
        self.current_request_id: Optional[str] = None
    
    def load_sample_files(self) -> List[str]:
        try:
            response = requests.get(f"{self.api_base_url}/api/samples")
            if response.ok:
                data = response.json()
                samples = []
                for sample in data.get("samples", []):
                    level = sample.get("level", "")
                    name = sample.get("name", "")
                    path = sample.get("path", "")
                    samples.append(f"[{level}] {name}")
                return ["-- Select a sample to load --"] + samples
            return ["-- Select a sample to load --"]
        except Exception as e:
            print(f"Error loading samples: {e}")
            return ["-- Select a sample to load --"]
    
    def load_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Load available prompt templates from API"""
        try:
            response = requests.get(f"{self.api_base_url}/api/prompts")
            if response.ok:
                data = response.json()
                return data.get("prompts", {})
            return {}
        except Exception as e:
            print(f"Error loading prompt templates: {e}")
            return {}
    
    def get_prompt_template_choices(self) -> Tuple[List[str], Dict[str, str], str, str]:
        """Get formatted dropdown choices for prompt templates.
        
        Returns:
            Tuple of (choices list, display_to_key mapping, default value, default content)
        """
        templates = self.load_prompt_templates()
        display_to_key = {}
        default_value = "-- Select template to load --"
        default_content = ""
        
        # Define the desired order
        ordered_keys = ["RDNA4_PROMPT", "HIGH_CORRECT_PROMPT", "HIGH_PERF_PROMPT", "QUANT_OP_PROMPT"]
        
        choices = ["-- Select template to load --"]
        for key in ordered_keys:
            if key in templates:
                template = templates[key]
                name = template.get("name", key)
                description = template.get("description", "")
                # Format: "Name - Description" for better readability
                display_label = f"{name} - {description}" if description else name
                choices.append(display_label)
                display_to_key[display_label] = key
                
                # Set RDNA4_PROMPT as default for RDNA4 deployments
                if key == "RDNA4_PROMPT":
                    default_value = display_label
                    default_content = template.get("content", "")
        
        # Add any remaining templates not in the ordered list
        for key, template in templates.items():
            if key not in ordered_keys:
                name = template.get("name", key)
                description = template.get("description", "")
                display_label = f"{name} - {description}" if description else name
                choices.append(display_label)
                display_to_key[display_label] = key
        
        return choices, display_to_key, default_value, default_content
    
    def load_sample_content(self, sample_selection: str) -> Tuple[str, str]:
        """Load sample content and return both content and problem name"""
        if sample_selection == "-- Select a sample to load --":
            return ("", "")
        
        try:
            # Dynamically parse level from "[levelX] name" format
            match = re.match(r'\[(level\d+)\]\s*(.+)', sample_selection)
            if not match:
                return ("", "")
            
            level = match.group(1)
            name = match.group(2).strip()
            
            filename = name + ".py"
            response = requests.get(f"{self.api_base_url}/api/samples/{level}/{filename}")
            if response.ok:
                data = response.json()
                content = data.get("content", "")
                # Return content and the problem name (without .py extension)
                return (content, name)
            return ("", "")
        except Exception as e:
            return (f"Error loading sample: {str(e)}", "")
    
    def update_model_name(self, server_type: str) -> str:
        model_map = {
            "deepseek": "deepseek-coder",
            "openai": "gpt-5",
            "anthropic": "claude-sonnet-4.5",
            "google": "gemini-1.5-flash-002",
            "nim": "qwen/qwen3-coder-480b-a35b-instruct"
        }
        return model_map.get(server_type, "gpt-5")
    
    def get_prompt_template_content(self, display_label: str, display_to_key: Dict[str, str]) -> str:
        """Get the content of a selected prompt template.
        
        Args:
            display_label: The display label shown in dropdown
            display_to_key: Mapping from display labels to template keys
        """
        if display_label == "-- Select template to load --":
            return gr.update()  # No change
        
        # Get the actual template key from display label
        template_key = display_to_key.get(display_label)
        if not template_key:
            return gr.update()
        
        templates = self.load_prompt_templates()
        if template_key in templates:
            return templates[template_key].get("content", "")
        return gr.update()  # No change
    
    def submit_generation(
        self,
        ref_arch_src: str,
        backend: str,
        server_type: str,
        model_name: str,
        gpu_arch: str,
        max_tokens: int,
        temperature: float,
        custom_prompt: str,
        problem_name: str,
        max_retries: int,
        target_speedup: float,
        progress=gr.Progress()
    ) -> Tuple[str, str, str, str]:
        if not ref_arch_src or not ref_arch_src.strip():
            return ("", "", "❌ Please provide reference architecture source code", "")
        
        try:
            request_data = {
                "ref_arch_src": ref_arch_src,
                "gpu_arch": [gpu_arch],
                "backend": backend,
                "model_name": model_name,
                "server_type": server_type,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "custom_prompt": custom_prompt if custom_prompt and custom_prompt.strip() else None,
                "problem_name": problem_name if problem_name and problem_name.strip() else None,
                "max_retries": int(max_retries) if max_retries is not None else None,
                "target_speedup": float(target_speedup)
            }
            
            response = requests.post(
                f"{self.api_base_url}/api/generate",
                json=request_data
            )
            
            if not response.ok:
                error_detail = response.json().get("detail", "Unknown error")
                return ("", "", f"❌ Failed to submit request: {error_detail}", "")
            
            result = response.json()
            request_id = result.get("request_id")
            self.current_request_id = request_id
            
            progress(0, desc="Request submitted, waiting for processing...")
            
            start_time = time.time()
            max_wait_time = 1200
            
            while True:
                if time.time() - start_time > max_wait_time:
                    return ("", "", f"⏱️ Request timed out (ID: {request_id[:8]}...)", "")
                
                status_response = requests.get(f"{self.api_base_url}/api/status/{request_id}")
                if not status_response.ok:
                    return ("", "", f"❌ Failed to check status for request {request_id[:8]}...", "")
                
                status_data = status_response.json()
                current_status = status_data.get("status")
                
                if current_status == "pending":
                    progress(0.25, desc="⏳ Request queued...")
                elif current_status == "processing":
                    elapsed = int(time.time() - start_time)
                    current_retry = status_data.get("current_retry", 0)
                    max_retries = status_data.get("max_retries", 0)
                    retry_history = status_data.get("retry_history", [])
                    
                    # Build progress description with latest evaluation results
                    desc_parts = []
                    if current_retry > 0:
                        desc_parts.append(f"🔄 Attempt {current_retry + 1}/{max_retries + 1}")
                    else:
                        desc_parts.append(f"🔄 Generating kernel")
                    
                    # Add latest evaluation summary if available
                    if retry_history:
                        latest = retry_history[-1]
                        eval_sum = latest.get('eval_summary')
                        if eval_sum:
                            compiled = "✅" if eval_sum.get('compiled') else "❌"
                            correct = "✅" if eval_sum.get('correctness') else "❌"
                            speedup = eval_sum.get('speedup', 0)
                            desc_parts.append(f"| Compiled:{compiled} Correct:{correct}")
                            if speedup > 0:
                                speedup_icon = "🚀" if speedup >= 1.0 else "⚠️"
                                desc_parts.append(f"Speedup:{speedup_icon}{speedup:.2f}x")
                    
                    desc_parts.append(f"({elapsed}s)")
                    progress(0.5, desc=" ".join(desc_parts))
                elif current_status == "completed":
                    progress(1.0, desc="✅ Generation completed!")
                    
                    generated_kernel = status_data.get("generated_kernel", "No kernel generated")
                    eval_result_str = status_data.get("eval_result", "")
                    eval_formatted = self.format_eval_results(eval_result_str)
                    
                    current_retry = status_data.get("current_retry", 0)
                    retry_info = f"\n**Attempts:** {current_retry + 1}" if current_retry > 0 else ""
                    success_msg = f"✅ Generation completed successfully!{retry_info}\n**Request ID:** `{request_id[:8]}...`"
                    
                    return (generated_kernel, eval_formatted, success_msg, request_id)
                    
                elif current_status == "failed":
                    error_msg = status_data.get("error_message", "Unknown error")
                    current_retry = status_data.get("current_retry", 0)
                    max_retries = status_data.get("max_retries", 0)
                    retry_info = f"\n**Total Attempts:** {current_retry + 1}/{max_retries + 1}" if max_retries > 0 else ""
                    return ("", "", f"❌ Generation failed: {error_msg}{retry_info}\n**Request ID:** `{request_id[:8]}...`", request_id)
                
                time.sleep(2)
                
        except Exception as e:
            return ("", "", f"❌ Error: {str(e)}", "")
    
    def format_eval_results(self, eval_result_str: str) -> str:
        if not eval_result_str:
            return "⚠️ No evaluation results available"
        
        try:
            result = self.parse_eval_string(eval_result_str)
            
            sections = []
            
            sections.append("### 📊 Final Evaluation Results\n")
            
            compiled_status = "✅ Compiled" if result.get("compiled") else "❌ Failed to Compile"
            correctness_status = "✅ Correct" if result.get("correctness") else "❌ Incorrect"
            sections.append(f"**Status:** {compiled_status} | {correctness_status}\n")
            
            metadata = result.get("metadata", {})
            if metadata:
                hardware = metadata.get("hardware", "Unknown")
                device = metadata.get("device", "")
                device_str = f" (Device {device})" if device else ""
                sections.append(f"**Hardware:** {hardware}{device_str}")
                
                correctness_trials = metadata.get("correctness_trials")
                if correctness_trials:
                    sections.append(f"**Correctness Trials:** {correctness_trials}")
            
            speedup = result.get("speedup")
            if speedup and speedup > 0:
                speedup_emoji = "🚀" if speedup > 1 else "⚠️" if speedup < 1 else "➖"
                speedup_desc = "Faster than reference!" if speedup > 1 else "Slower than reference" if speedup < 1 else "Same as reference"
                sections.append(f"\n**Speedup:** {speedup_emoji} **{speedup:.2f}x** - {speedup_desc}")
            
            ref_runtime = result.get("ref_runtime")
            runtime = result.get("runtime")
            
            if ref_runtime is not None and runtime is not None:
                sections.append("\n### Performance Comparison\n")
                
                ref_stats = result.get("ref_runtime_stats", {})
                runtime_stats = result.get("runtime_stats", {})
                
                sections.append("| Model | Mean Runtime | Std Dev | Min | Max |")
                sections.append("|-------|--------------|---------|-----|-----|")
                sections.append(
                    f"| **Reference (PyTorch)** | {ref_runtime:.2f} ms | "
                    f"{ref_stats.get('std', 0):.4f} ms | "
                    f"{ref_stats.get('min', 0):.2f} ms | "
                    f"{ref_stats.get('max', 0):.2f} ms |"
                )
                sections.append(
                    f"| **Custom Kernel** | {runtime:.2f} ms | "
                    f"{runtime_stats.get('std', 0):.4f} ms | "
                    f"{runtime_stats.get('min', 0):.2f} ms | "
                    f"{runtime_stats.get('max', 0):.2f} ms |"
                )
                
                num_trials = runtime_stats.get("num_trials", "N/A")
                sections.append(f"\n*Number of trials: {num_trials}*")
            elif runtime is not None:
                sections.append(f"\n**Runtime:** {runtime:.2f} ms")
                
                runtime_stats = result.get("runtime_stats", {})
                if runtime_stats:
                    sections.append("\n### Runtime Statistics\n")
                    sections.append("| Metric | Value |")
                    sections.append("|--------|-------|")
                    sections.append(f"| Mean | {runtime_stats.get('mean', 0):.2f} ms |")
                    sections.append(f"| Std Dev | {runtime_stats.get('std', 0):.4f} ms |")
                    sections.append(f"| Min | {runtime_stats.get('min', 0):.2f} ms |")
                    sections.append(f"| Max | {runtime_stats.get('max', 0):.2f} ms |")
                    sections.append(f"| Trials | {runtime_stats.get('num_trials', 'N/A')} |")
            
            return "\n".join(sections)
            
        except Exception as e:
            return f"⚠️ Error formatting results:\n```\n{eval_result_str}\n```"
    
    def format_attempt_history(self, retry_history: List[Dict]) -> str:
        """Format attempt history for display"""
        if not retry_history:
            return ""
        
        sections = ["\n\n---\n\n### 🔄 Attempt History\n"]
        
        for entry in retry_history:
            attempt_num = entry.get('attempt', 0) + 1
            eval_sum = entry.get('eval_summary')
            
            if not eval_sum:
                continue
            
            compiled = eval_sum.get('compiled', False)
            correctness = eval_sum.get('correctness', False)
            speedup = eval_sum.get('speedup', 0)
            runtime = eval_sum.get('runtime', 0)
            ref_runtime = eval_sum.get('ref_runtime', 0)
            
            # Status icons
            compiled_icon = "✅" if compiled else "❌"
            correct_icon = "✅" if correctness else "❌"
            
            sections.append(f"\n**Attempt {attempt_num}:**")
            sections.append(f"- Compiled: {compiled_icon}")
            sections.append(f"- Correctness: {correct_icon}")
            
            if compiled and correctness and speedup > 0:
                speedup_icon = "🚀" if speedup >= 1.0 else "⚠️"
                sections.append(f"- Speedup: {speedup_icon} **{speedup:.3f}x**")
                sections.append(f"- Runtime: {runtime:.2f} ms (ref: {ref_runtime:.2f} ms)")
            elif entry.get('error'):
                error_msg = entry.get('error', '')[:200]
                sections.append(f"- Issue: {error_msg}...")
        
        return "\n".join(sections)
    
    def parse_eval_string(self, s: str) -> Dict[str, Any]:
        import re
        import json
        
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
        
        try:
            compiled_match = re.search(r'compiled=(True|False)', s)
            if compiled_match:
                result["compiled"] = compiled_match.group(1) == "True"
            
            correctness_match = re.search(r'correctness=(True|False)', s)
            if correctness_match:
                result["correctness"] = correctness_match.group(1) == "True"
            
            metadata_match = re.search(r'metadata=(\{[^}]*\})', s)
            if metadata_match:
                try:
                    metadata_str = metadata_match.group(1).replace("'", '"')
                    metadata_str = re.sub(r'\((\d+)\s*/\s*(\d+)\)', r'"(\1 / \2)"', metadata_str)
                    result["metadata"] = json.loads(metadata_str)
                except:
                    pass
            
            runtime_match = re.search(r'\bruntime=([\d.]+)(?=\s|$|runtime_stats|ref_)', s)
            if runtime_match:
                result["runtime"] = float(runtime_match.group(1))
            
            ref_runtime_match = re.search(r'ref_runtime=([\d.]+)', s)
            if ref_runtime_match:
                result["ref_runtime"] = float(ref_runtime_match.group(1))
            
            speedup_match = re.search(r'speedup=([\d.]+)', s)
            if speedup_match:
                result["speedup"] = float(speedup_match.group(1))
            
            def extract_dict(pattern: str, start_str: str) -> Optional[Dict]:
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
            
            runtime_stats = extract_dict(r'runtime_stats=', 'runtime_stats=')
            if runtime_stats:
                result["runtime_stats"] = runtime_stats
            
            ref_runtime_stats = extract_dict(r'ref_runtime_stats=', 'ref_runtime_stats=')
            if ref_runtime_stats:
                result["ref_runtime_stats"] = ref_runtime_stats
                
        except Exception as e:
            print(f"Error parsing eval string: {e}")
        
        return result
    
    def load_request_history(self, limit: int = None) -> Tuple[List[List[Any]], List[str]]:
        try:
            # Request all records by setting a very high limit
            # The API will return all available requests
            url = f"{self.api_base_url}/api/requests?limit=10000"
            response = requests.get(url)
            if not response.ok:
                return ([], [])
            
            data = response.json()
            requests_list = data.get("requests", [])
            
            if not requests_list:
                return ([], [])
            
            table_data = []
            request_ids = []
            
            for req in requests_list:
                req_id = req.get("id", "")
                request_ids.append(req_id)
                
                req_id_short = req_id[:12] + "..."
                status = req.get("status", "unknown")
                backend = req.get("backend", "-")
                model_name = req.get("model_name", "-")
                problem_name = req.get("problem_name", "-")
                created_at = req.get("created_at", "")
                
                compiled = "-"
                correctness = "-"
                runtime = "-"
                speedup = "-"
                
                if req.get("eval_result") and status == "completed":
                    try:
                        eval_data = self.parse_eval_string(req.get("eval_result", ""))
                        
                        if eval_data.get("compiled") is not None:
                            compiled = "✅" if eval_data["compiled"] else "❌"
                        
                        if eval_data.get("correctness") is not None:
                            correctness = "✅" if eval_data["correctness"] else "❌"
                        
                        if eval_data.get("runtime") is not None:
                            runtime = f"{eval_data['runtime']:.2f}"
                        
                        if eval_data.get("speedup") is not None and eval_data["speedup"] > 0:
                            speedup = f"{eval_data['speedup']:.2f}x"
                    except:
                        pass
                
                status_emoji = {
                    "pending": "⏳",
                    "processing": "🔄",
                    "completed": "✅",
                    "failed": "❌"
                }.get(status, "❓")
                
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    created_str = dt.strftime("%m/%d %H:%M")
                except:
                    created_str = created_at[:16] if len(created_at) > 16 else created_at
                
                table_data.append([
                    req_id_short,
                    f"{status_emoji} {status}",
                    backend,
                    model_name,
                    problem_name,
                    compiled,
                    correctness,
                    runtime,
                    speedup,
                    created_str
                ])
            
            return (table_data, request_ids)
            
        except Exception as e:
            print(f"Error loading history: {e}")
            return ([], [])
    
    def view_request_by_id(self, request_id: str) -> Tuple[str, str, str, str, str]:
        if not request_id or not request_id.strip():
            return ("", "", "", "", "⚠️ Please enter a request ID")
        
        try:
            response = requests.get(f"{self.api_base_url}/api/status/{request_id}")
            if not response.ok:
                return ("", "", "", "", f"❌ Request not found: {request_id}")
            
            status_data = response.json()
            status = status_data.get("status")
            
            if status == "completed":
                ref_code = status_data.get("ref_arch_src", "No reference code")
                generated_kernel = status_data.get("generated_kernel", "No kernel generated")
                
                # Format evaluation results with attempt history
                eval_result = self.format_eval_results(status_data.get("eval_result", ""))
                retry_history = status_data.get("retry_history", [])
                if retry_history:
                    eval_result += self.format_attempt_history(retry_history)
                
                error_message = status_data.get("error_message", "")
                msg = f"✅ Request `{request_id[:12]}...` loaded successfully"
                return (ref_code, generated_kernel, eval_result, error_message, msg)
            elif status == "failed":
                # For failed requests, still load ref_code, generated_kernel (last attempt), and error
                ref_code = status_data.get("ref_arch_src", "No reference code")
                generated_kernel = status_data.get("generated_kernel", "No kernel generated (all attempts failed)")
                error_msg = status_data.get("error_message", "Unknown error")
                eval_result = self.format_eval_results(status_data.get("eval_result", "")) if status_data.get("eval_result") else ""
                
                # Show retry information if available
                current_retry = status_data.get("current_retry", 0)
                max_retries = status_data.get("max_retries", 0)
                retry_info = f"\n\n**Total Attempts:** {current_retry + 1}/{max_retries + 1}" if max_retries > 0 else ""
                
                msg = f"❌ Request failed after all attempts{retry_info}"
                return (ref_code, generated_kernel, eval_result, error_msg, msg)
            else:
                return ("", "", "", "", f"⏳ Request is still {status}")
                
        except Exception as e:
            return ("", "", "", "", f"❌ Error: {str(e)}")
    
    def view_request_from_table(self, history_table_data: gr.SelectData, request_ids_state: List[str]) -> Tuple[str, str, str, str, str]:
        try:
            row_index = history_table_data.index[0]
            if row_index < len(request_ids_state):
                request_id = request_ids_state[row_index]
                return self.view_request_by_id(request_id)
            return ("", "", "", "", "⚠️ Invalid selection")
        except Exception as e:
            return ("", "", "", "", f"❌ Error: {str(e)}")
    
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="TritonAgent - GPU Kernel Generator", theme=gr.themes.Base()) as app:
            gr.Markdown(
                """
                # 🚀 TritonAgent GPU Kernel Generator
                **Powered by AMD** - Generate and evaluate optimized GPU kernels
                
                📖 **[User Guide](/guide)** - Learn how to use TritonAgent effectively
                """
            )
            
            with gr.Tabs():
                with gr.Tab("Generate Kernel"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Input Configuration")
                            
                            sample_dropdown = gr.Dropdown(
                                choices=self.load_sample_files(),
                                label="Load Sample Code",
                                value="-- Select a sample to load --",
                                interactive=True
                            )
                            
                            ref_arch_src = gr.Textbox(
                                label="Reference Architecture Source Code",
                                placeholder="Paste your PyTorch reference implementation here...",
                                lines=12,
                                max_lines=20
                            )
                            
                            # Advanced Settings - Custom Prompt with Template Loader
                            with gr.Accordion("Advanced Settings", open=False):
                                template_choices, display_to_key_map, default_template, default_content = self.get_prompt_template_choices()
                                display_to_key_state = gr.State(display_to_key_map)
                                
                                prompt_template_btn = gr.Dropdown(
                                    choices=template_choices,
                                    value=default_template,
                                    label="Load Prompt Template",
                                    info="Select a template to load into custom prompt (you can edit after loading)"
                                )
                                
                                custom_prompt_input = gr.Textbox(
                                    label="Custom Prompt (Optional)",
                                    placeholder="Add custom instructions or load a template above...",
                                    value=default_content,
                                    lines=5,
                                    max_lines=10
                                )
                            
                            problem_name_state = gr.State("")
                            
                            with gr.Row():
                                backend = gr.Dropdown(
                                    choices=["triton"],
                                    value="triton",
                                    label="Backend"
                                )
                                
                                server_type = gr.Dropdown(
                                    choices=["deepseek", "openai", "anthropic", "google", "nim"],
                                    value="anthropic",
                                    label="Model Provider"
                                )
                            
                            with gr.Row():
                                model_name = gr.Textbox(
                                    label="Model Name",
                                    value="claude-sonnet-4.5"
                                )
                                
                                gpu_arch = gr.Dropdown(
                                    choices=["CDNA", "RDNA"],
                                    value="CDNA",
                                    label="GPU Architecture"
                                )
                            
                            with gr.Row():
                                max_tokens = gr.Number(
                                    label="Max Tokens",
                                    value=4096,
                                    minimum=256,
                                    maximum=8192
                                )
                                
                                temperature = gr.Slider(
                                    label="Temperature",
                                    minimum=1.0,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.1
                                )
                            
                            with gr.Row():
                                max_retries = gr.Number(
                                    label="Max Retries (Reflection)",
                                    value=3,
                                    minimum=0,
                                    maximum=10,
                                    info="Number of times to retry generation with error feedback"
                                )
                                
                                target_speedup = gr.Slider(
                                    label="Target Speedup",
                                    value=1.5,
                                    minimum=0.1,
                                    maximum=10.0,
                                    step=0.1,
                                    info="Target speedup threshold for early exit (1.0 = same as baseline)"
                                )
                            
                            generate_btn = gr.Button("🚀 Generate Kernel", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### Results")
                            
                            status_msg = gr.Markdown("")
                            
                            generated_kernel = gr.Code(
                                label="Generated Kernel",
                                language="cpp",
                                lines=15
                            )
                            
                            eval_results = gr.Markdown(
                                label="Evaluation Results"
                            )
                            
                            request_id_state = gr.State("")
                    
                    sample_dropdown.change(
                        fn=self.load_sample_content,
                        inputs=[sample_dropdown],
                        outputs=[ref_arch_src, problem_name_state]
                    )
                    
                    prompt_template_btn.change(
                        fn=self.get_prompt_template_content,
                        inputs=[prompt_template_btn, display_to_key_state],
                        outputs=[custom_prompt_input]
                    )
                    
                    server_type.change(
                        fn=self.update_model_name,
                        inputs=[server_type],
                        outputs=[model_name]
                    )
                    
                    generate_btn.click(
                        fn=self.submit_generation,
                        inputs=[
                            ref_arch_src, backend, server_type, model_name,
                            gpu_arch, max_tokens, temperature, custom_prompt_input, 
                            problem_name_state, max_retries, target_speedup
                        ],
                        outputs=[generated_kernel, eval_results, status_msg, request_id_state],
                        # Enable queue for long-running tasks (kernel generation can take several minutes)
                        # Allow multiple concurrent requests to match backend capability
                        concurrency_limit=10  # Allow up to 10 concurrent executions
                    )
                
                with gr.Tab("Request History"):
                    gr.Markdown("### All Generation Requests")
                    gr.Markdown("💡 **Tip:** Click on any row in the table to view its details below")
                    
                    request_ids_state = gr.State([])
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                    
                    history_table = gr.Dataframe(
                        headers=["ID", "Status", "Backend", "Model", "Problem", "Compiled", "Correct", "Runtime", "Speedup", "Created"],
                        datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### Request Details")
                    
                    view_status_msg = gr.Markdown("")
                    
                    with gr.Tabs():
                        with gr.Tab("Reference Code"):
                            view_ref_code = gr.Code(
                                label="Reference Architecture Source Code",
                                language="python",
                                lines=12
                            )
                        
                        with gr.Tab("Generated Kernel"):
                            view_kernel_output = gr.Code(
                                label="Generated Kernel Code",
                                language="cpp",
                                lines=12
                            )
                        
                        with gr.Tab("Evaluation Results"):
                            view_eval_output = gr.Markdown(
                                label="Evaluation Results"
                            )
                        
                        with gr.Tab("Error Message"):
                            view_error_message = gr.Textbox(
                                label="Error Details",
                                lines=12,
                                max_lines=100,
                                interactive=False,
                                placeholder="No error message available"
                            )
                    
                    def refresh_history():
                        table_data, req_ids = self.load_request_history()
                        return table_data, req_ids
                    
                    refresh_btn.click(
                        fn=refresh_history,
                        inputs=[],
                        outputs=[history_table, request_ids_state]
                    )
                    
                    history_table.select(
                        fn=self.view_request_from_table,
                        inputs=[request_ids_state],
                        outputs=[view_ref_code, view_kernel_output, view_eval_output, view_error_message, view_status_msg]
                    )
                    
                    app.load(
                        fn=refresh_history,
                        inputs=[],
                        outputs=[history_table, request_ids_state]
                    )
            
            # gr.Markdown(
            #     """
            #     ---
            #     **Note:** This interface provides the same functionality as the web frontend.
            #     Backend API must be running at `http://localhost:8009` for this to work.
            #     """
            # )
        
        return app


def create_gradio_app(api_base_url: str = "http://localhost:8009") -> gr.Blocks:
    app_instance = KernelBenchGradioApp(api_base_url=api_base_url)
    return app_instance.create_interface()


def serve_guide_html() -> str:
    """Read and return the guide HTML content."""
    guide_path = os.path.join(os.path.dirname(__file__), "templates", "guide.html")
    try:
        with open(guide_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<html><body><h1>Guide not found</h1><p>The user guide file is missing.</p></body></html>"


def create_app() -> FastAPI:
    """Create the combined FastAPI + Gradio application."""
    api_url = os.environ.get("API_BASE_URL", "http://localhost:8009")
    gradio_app = create_gradio_app(api_base_url=api_url)
    gradio_app.queue(max_size=20)
    
    fastapi_app = FastAPI(title="TritonAgent")
    
    @fastapi_app.get("/guide", response_class=HTMLResponse)
    async def get_guide():
        """Serve the user guide page."""
        return HTMLResponse(content=serve_guide_html())
    
    return gr.mount_gradio_app(fastapi_app, gradio_app, path="/")


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=7862)
