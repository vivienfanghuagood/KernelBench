import gradio as gr
import requests
import time
import os
from typing import Optional, Dict, Any, List, Tuple

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
    
    def load_sample_content(self, sample_selection: str) -> str:
        if sample_selection == "-- Select a sample to load --":
            return ""
        
        try:
            if sample_selection.startswith("[level1]"):
                level = "level1"
                name = sample_selection[9:].strip()
            elif sample_selection.startswith("[level2]"):
                level = "level2"
                name = sample_selection[9:].strip()
            else:
                return ""
            
            filename = name + ".py"
            response = requests.get(f"{self.api_base_url}/api/samples/{level}/{filename}")
            if response.ok:
                data = response.json()
                return data.get("content", "")
            return ""
        except Exception as e:
            return f"Error loading sample: {str(e)}"
    
    def update_model_name(self, server_type: str) -> str:
        model_map = {
            "deepseek": "deepseek-coder",
            "openai": "gpt-5",
            "anthropic": "claude-3-5-sonnet-20241022",
            "google": "gemini-1.5-flash-002"
        }
        return model_map.get(server_type, "gpt-5")
    
    def submit_generation(
        self,
        ref_arch_src: str,
        backend: str,
        server_type: str,
        model_name: str,
        gpu_arch: str,
        max_tokens: int,
        temperature: float,
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
                "temperature": float(temperature)
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
            max_wait_time = 300
            
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
                    progress(0.5, desc=f"🔄 Generating kernel... ({elapsed}s elapsed)")
                elif current_status == "completed":
                    progress(1.0, desc="✅ Generation completed!")
                    
                    generated_kernel = status_data.get("generated_kernel", "No kernel generated")
                    eval_result_str = status_data.get("eval_result", "")
                    eval_formatted = self.format_eval_results(eval_result_str)
                    
                    success_msg = f"✅ Generation completed successfully!\n**Request ID:** `{request_id[:8]}...`"
                    
                    return (generated_kernel, eval_formatted, success_msg, request_id)
                    
                elif current_status == "failed":
                    error_msg = status_data.get("error_message", "Unknown error")
                    return ("", "", f"❌ Generation failed: {error_msg}\n**Request ID:** `{request_id[:8]}...`", request_id)
                
                time.sleep(2)
                
        except Exception as e:
            return ("", "", f"❌ Error: {str(e)}", "")
    
    def format_eval_results(self, eval_result_str: str) -> str:
        if not eval_result_str:
            return "⚠️ No evaluation results available"
        
        try:
            result = self.parse_eval_string(eval_result_str)
            
            sections = []
            
            sections.append("### Evaluation Results\n")
            
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
    
    def load_request_history(self, limit: int = 10) -> Tuple[List[List[Any]], List[str]]:
        try:
            response = requests.get(f"{self.api_base_url}/api/requests?limit={limit}")
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
    
    def view_request_by_id(self, request_id: str) -> Tuple[str, str, str, str]:
        if not request_id or not request_id.strip():
            return ("", "", "", "⚠️ Please enter a request ID")
        
        try:
            response = requests.get(f"{self.api_base_url}/api/status/{request_id}")
            if not response.ok:
                return ("", "", "", f"❌ Request not found: {request_id}")
            
            status_data = response.json()
            status = status_data.get("status")
            
            if status == "completed":
                ref_code = status_data.get("ref_arch_src", "No reference code")
                generated_kernel = status_data.get("generated_kernel", "No kernel generated")
                eval_result = self.format_eval_results(status_data.get("eval_result", ""))
                msg = f"✅ Request `{request_id[:12]}...` loaded successfully"
                return (ref_code, generated_kernel, eval_result, msg)
            elif status == "failed":
                error_msg = status_data.get("error_message", "Unknown error")
                return ("", "", "", f"❌ Request failed: {error_msg}")
            else:
                return ("", "", "", f"⏳ Request is still {status}")
                
        except Exception as e:
            return ("", "", "", f"❌ Error: {str(e)}")
    
    def view_request_from_table(self, history_table_data: gr.SelectData, request_ids_state: List[str]) -> Tuple[str, str, str, str]:
        try:
            row_index = history_table_data.index[0]
            if row_index < len(request_ids_state):
                request_id = request_ids_state[row_index]
                return self.view_request_by_id(request_id)
            return ("", "", "", "⚠️ Invalid selection")
        except Exception as e:
            return ("", "", "", f"❌ Error: {str(e)}")
    
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="KernelBench - GPU Kernel Generator", theme=gr.themes.Soft()) as app:
            gr.Markdown(
                """
                # 🚀 KernelBench GPU Kernel Generator
                **Powered by AMD** - Generate and evaluate optimized GPU kernels
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
                            
                            with gr.Row():
                                backend = gr.Dropdown(
                                    choices=["cuda", "triton", "cute"],
                                    value="triton",
                                    label="Backend"
                                )
                                
                                server_type = gr.Dropdown(
                                    choices=["deepseek", "openai", "anthropic", "google"],
                                    value="openai",
                                    label="Model Provider"
                                )
                            
                            with gr.Row():
                                model_name = gr.Textbox(
                                    label="Model Name",
                                    value="gpt-5"
                                )
                                
                                gpu_arch = gr.Dropdown(
                                    choices=["mi300", "Ada", "Hopper"],
                                    value="Ada",
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
                        outputs=[ref_arch_src]
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
                            gpu_arch, max_tokens, temperature
                        ],
                        outputs=[generated_kernel, eval_results, status_msg, request_id_state]
                    )
                
                with gr.Tab("Request History"):
                    gr.Markdown("### Recent Generation Requests")
                    gr.Markdown("💡 **Tip:** Click on any row in the table to view its details below")
                    
                    request_ids_state = gr.State([])
                    
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                        limit_slider = gr.Slider(
                            label="Number of requests",
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=5
                        )
                    
                    history_table = gr.Dataframe(
                        headers=["ID", "Status", "Backend", "Model", "Compiled", "Correct", "Runtime", "Speedup", "Created"],
                        datatype=["str", "str", "str", "str", "str", "str", "str", "str", "str"],
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
                    
                    def refresh_history(limit):
                        table_data, req_ids = self.load_request_history(limit)
                        return table_data, req_ids
                    
                    refresh_btn.click(
                        fn=refresh_history,
                        inputs=[limit_slider],
                        outputs=[history_table, request_ids_state]
                    )
                    
                    limit_slider.change(
                        fn=refresh_history,
                        inputs=[limit_slider],
                        outputs=[history_table, request_ids_state]
                    )
                    
                    history_table.select(
                        fn=self.view_request_from_table,
                        inputs=[request_ids_state],
                        outputs=[view_ref_code, view_kernel_output, view_eval_output, view_status_msg]
                    )
                    
                    app.load(
                        fn=refresh_history,
                        inputs=[limit_slider],
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


if __name__ == "__main__":
    api_url = os.environ.get("API_BASE_URL", "http://localhost:8009")
    app = create_gradio_app(api_base_url=api_url)
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
