import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.gradio_app import create_gradio_app

if __name__ == "__main__":
    api_url = os.environ.get("API_BASE_URL", "http://localhost:8009")
    
    print("=" * 60)
    print("Starting KernelBench Gradio Interface")
    print("=" * 60)
    print(f"API Base URL: {api_url}")
    print("Make sure the FastAPI backend is running at the API URL!")
    print("=" * 60)
    
    app = create_gradio_app(api_base_url=api_url)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
