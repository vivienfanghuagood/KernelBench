# KernelBench API

A FastAPI service for generating and evaluating GPU kernels using the KernelBench framework.

## Features

- **Async Kernel Generation**: Submit requests and poll for results
- **Multiple Backends**: Support for CUDA, Triton, and CuTe
- **Multiple Models**: DeepSeek, OpenAI, Anthropic, Google
- **Web Interface**: Clean frontend for easy interaction
- **Request History**: Track and view previous generations
- **SQLite Persistence**: Local database for request storage

## Quick Start

### 1. Install Dependencies

```bash
# Install API-specific requirements
cd api
pip install -r requirements.txt

# Make sure main KernelBench dependencies are installed
cd ..
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```bash
# DeepSeek (default)
DEEPSEEK_API_KEY=your_deepseek_key

# OpenAI (optional)
OPENAI_API_KEY=your_openai_key

# Anthropic (optional)  
ANTHROPIC_API_KEY=your_anthropic_key

# Google (optional)
GEMINI_API_KEY=your_gemini_key
```

### 3. Run the API Server

```bash
cd api
python main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Access the Interface

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### POST /api/generate

Submit a new kernel generation request.

**Request Body:**
```json
{
  "ref_arch_src": "import torch\n\ndef my_function(x, y):\n    return torch.matmul(x, y)",
  "gpu_arch": ["Ada"],
  "backend": "cuda",
  "model_name": "deepseek-coder",
  "server_type": "deepseek",
  "max_tokens": 4096,
  "temperature": 0.0
}
```

**Response:**
```json
{
  "request_id": "uuid-string",
  "status": "pending",
  "message": "Generation request submitted successfully"
}
```

### GET /api/status/{request_id}

Check the status of a generation request.

**Response:**
```json
{
  "request_id": "uuid-string",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00",
  "started_at": "2024-01-01T12:00:01",
  "completed_at": "2024-01-01T12:01:30",
  "generated_kernel": "// Generated CUDA kernel code...",
  "eval_result": "Evaluation results...",
  "error_message": null
}
```

### GET /api/requests

Get all recent generation requests.

## Status Values

- **pending**: Request submitted, waiting to start
- **processing**: Currently generating kernel
- **completed**: Generation finished successfully
- **failed**: Generation failed with error

## Supported Backends

- **cuda**: Generate CUDA kernels
- **triton**: Generate Triton kernels  
- **cute**: Generate CuTe kernels

## Supported Model Providers

- **deepseek**: DeepSeek models (default: deepseek-coder)
- **openai**: OpenAI models (default: gpt-4o-2024-08-06)
- **anthropic**: Anthropic models (default: claude-3-5-sonnet-20241022)
- **google**: Google models (default: gemini-1.5-flash-002)

## Web Interface Usage

1. **Paste Reference Code**: Copy your PyTorch reference implementation
2. **Configure Parameters**: Select backend, model, GPU architecture
3. **Submit Request**: Click "Generate Kernel" to start
4. **Monitor Progress**: Watch real-time status updates
5. **View Results**: See generated kernel and evaluation results
6. **Browse History**: Check previous requests in the table

## Development

### Project Structure

```
api/
├── main.py              # FastAPI application
├── service.py           # Kernel generation service
├── database.py          # SQLite database operations
├── requirements.txt     # API dependencies
├── static/
│   └── app.js          # Frontend JavaScript
└── templates/
    └── index.html      # Frontend HTML
```

### Running in Development

```bash
# With auto-reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# With custom log level
uvicorn api.main:app --log-level debug
```

### Database

The API uses SQLite for persistence. The database file `kernelbench_api.db` will be created automatically in the working directory.

### Error Handling

- API errors return appropriate HTTP status codes
- Frontend displays error messages to users
- Failed generations are logged with error details
- Database operations are wrapped in try-catch blocks

## Production Deployment

For production deployment, consider:

1. **Use PostgreSQL** instead of SQLite for better concurrency
2. **Add Authentication** for secure access
3. **Set up Redis** for request queueing
4. **Use Docker** for containerization
5. **Add Rate Limiting** to prevent abuse
6. **Set up Monitoring** for system health

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the KernelBench root directory
2. **Database Locked**: Stop all running instances before restarting
3. **GPU Architecture**: Ensure your GPU supports the selected architecture
4. **API Keys**: Check that environment variables are set correctly

### Logs

Check the console output for detailed error messages and generation progress.

### Database Reset

To reset the database:
```bash
rm kernelbench_api.db
python -c "from api.database import db; db.init_database()"
```