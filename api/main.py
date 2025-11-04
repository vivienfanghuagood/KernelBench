from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os

from api.service import kernel_service
from api.database import GenerationStatus

app = FastAPI(title="KernelBench API", description="API for generating and evaluating GPU kernels", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

class GenerationRequest(BaseModel):
    ref_arch_src: str
    gpu_arch: List[str] = ["4090"]
    backend: str = "triton"
    model_name: str = "gpt-5"
    server_type: str = "openai"
    max_tokens: int = 4096
    temperature: float = 0.0

class GenerationResponse(BaseModel):
    request_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    request_id: str
    status: str
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    generated_kernel: Optional[str]
    eval_result: Optional[str]
    error_message: Optional[str]

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    try:
        with open("api/templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>KernelBench API</title></head>
            <body>
                <h1>KernelBench API</h1>
                <p>Frontend template not found. Please check the templates directory.</p>
                <p><a href="/docs">View API documentation</a></p>
            </body>
        </html>
        """)

@app.post("/api/generate", response_model=GenerationResponse)
async def create_generation_request(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Submit a new kernel generation request"""
    try:
        # Validate backend
        if request.backend not in ["cuda", "triton", "cute"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported backend: {request.backend}. Must be 'cuda', 'triton', or 'cute'."
            )
        
        # Validate GPU architecture
        # valid_archs = ["mi300", "Ada", "Hopper"]
        # for arch in request.gpu_arch:
        #     if arch not in valid_archs:
        #         raise HTTPException(
        #             status_code=400,
        #             detail=f"Invalid GPU architecture: {arch}. Must be one of {valid_archs}"
        #         )
        
        # Submit the request
        request_id = kernel_service.submit_generation_request(
            ref_arch_src=request.ref_arch_src,
            gpu_arch=request.gpu_arch,
            backend=request.backend,
            model_name=request.model_name,
            server_type=request.server_type,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return GenerationResponse(
            request_id=request_id,
            status="pending",
            message="Generation request submitted successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{request_id}", response_model=StatusResponse)
async def get_generation_status(request_id: str):
    """Get the status of a generation request"""
    try:
        request_data = kernel_service.get_request_status(request_id)
        
        if not request_data:
            raise HTTPException(status_code=404, detail="Request not found")
        
        return StatusResponse(
            request_id=request_data['id'],
            status=request_data['status'],
            created_at=request_data['created_at'],
            started_at=request_data['started_at'],
            completed_at=request_data['completed_at'],
            generated_kernel=request_data['generated_kernel'],
            eval_result=request_data['eval_result'],
            error_message=request_data['error_message']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/requests")
async def get_all_requests(limit: int = 100):
    """Get all generation requests"""
    try:
        requests = kernel_service.get_all_requests(limit)
        return {"requests": requests}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "KernelBench API is running"}

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8009,
        reload=False,
        log_level="info"
    )