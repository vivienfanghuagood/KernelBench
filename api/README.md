# 🚀 TritonAgent API

Generate optimized Triton GPU kernels from PyTorch code with AI-powered agents.

## What is TritonAgent?

TritonAgent is an AI-powered tool designed specifically for **AMD Instinct GPUs**. It automatically converts your PyTorch code into optimized Triton GPU kernels, handling the complex work of writing high-performance GPU code for you.

```
📝 PyTorch Code → 🤖 Code Generation → ⚙️ Compile & Test → 📊 Benchmark → 🚀 Optimized Kernel
```

**The system automatically:**
- Generates Triton kernel code from your PyTorch model
- Validates correctness against the original implementation
- Measures speedup on AMD GPUs
- Retries with error feedback if compilation or tests fail

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
# Anthropic (Recommended)
ANTHROPIC_API_KEY=your_anthropic_key

# OpenAI (optional)
OPENAI_API_KEY=your_openai_key

# NIM (optional)
NIM_API_KEY=your_nim_key

# DeepSeek (optional)
DEEPSEEK_API_KEY=your_deepseek_key

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
- **User Guide**: http://localhost:8000/guide
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Input Code Format

Your input code needs three parts for the system to work:

### ① Model Class

A PyTorch module named `Model` with your implementation:

```python
class Model(nn.Module):
    def __init__(self, out_features):
        super(Model, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        return F.gelu(x[:, :self.out_features]) * x[:, self.out_features:]
```

### ② get_inputs() Function

Returns a list of input tensors for testing:

```python
def get_inputs():
    return [torch.rand(batch_size, features, dtype=torch.float16, device='cuda')]
```

### ③ get_init_inputs() Function

Returns arguments for creating the Model:

```python
def get_init_inputs():
    return [out_features]
```

> **Two input types supported:**
> - **Pure PyTorch** — Standard PyTorch ops (level1-6)
> - **PyTorch + Triton** — Already has Triton kernels to optimize (level7)

## Sample Categories

| Level | Description |
|-------|-------------|
| level1 | Basic operations — matrix multiplication, activations (ReLU, GELU, Softmax), norms |
| level2 | Simple fused operations — combining 2-3 basic ops |
| level3 | More complex fusions — multi-step computations |
| level4 | Transformer models — full attention and MLP blocks |
| **level6** 🆕 | Real-world operators from **SGLang** — production inference ops |
| **level7** 🆕 | Samples with existing **Triton forward** — optimize existing kernels |

## Configuration Options

### AI Model Providers

| Provider | Default Model | Status |
|----------|--------------|--------|
| Anthropic | `claude-sonnet-4.5` | ✅ Available |
| OpenAI | `gpt-5` | ✅ Available |
| NIM | `qwen/qwen3-coder-480b-a35b-instruct` | ✅ Available |
| DeepSeek | `deepseek-coder` | 🔜 Coming Soon |
| Google | `gemini-1.5-flash-002` | 🔜 Coming Soon |

See [llm.amd.com](https://llm.amd.com) for more model options.

### Other Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Backend | Triton | Kernel language (Triton, CUDA, CuTe) |
| GPU Architecture | CDNA | AMD Instinct GPUs (RDNA coming soon) |
| Max Retries | 3 | Auto-retry attempts on failure |
| Target Speedup | 1.0 | Stop early if this speedup is achieved |

## Viewing Results

After generation completes, you'll see the results immediately. You can also view past requests in the **Request History** tab.

### Understanding the Results

| Field | Meaning |
|-------|---------|
| Compiled ✅/❌ | Did the code compile without errors? |
| Correct ✅/❌ | Does the output match the original PyTorch? |
| Runtime | Execution time in milliseconds |
| Speedup | How much faster than PyTorch (e.g., 2.5x = 2.5 times faster) |

### Request History Tab

Click any row in the history table to view its details:
- **Reference Code** — Your original input
- **Generated Kernel** — The Triton code (copy this for your project!)
- **Evaluation Results** — Detailed performance metrics
- **Error Message** — Debug info if something failed

## Advanced: Custom Prompts

Custom prompts let you guide the AI with specific instructions. Use them when you need more control over the generated code.

### Built-in Templates

| Template | When to Use |
|----------|-------------|
| **QUANT_OP_PROMPT** | FP8/quantized ops, AMD hardware optimization |
| **HIGH_CORRECT_PROMPT** | Prioritize correctness over speed |
| **HIGH_PERF_PROMPT** | Maximize speedup (>2x target) |

Select a template from the dropdown, then modify it as needed. You can also write your own prompts from scratch.

### Troubleshooting Failed Generations

If generation fails or produces incorrect results:

1. Check the **Error Message** tab to understand what went wrong
2. Load **HIGH_CORRECT_PROMPT** template for correctness issues
3. Increase **Max Retries** to 5-10 for complex problems
4. Re-run the generation

> **Common fixes to add to your prompt:**
> - "Add @triton.jit decorator to all kernel functions"
> - "Use .to(dtype) instead of tl.astype()"
> - "Cast to fp32 for exp/log/sqrt operations"
> - "Keep BLOCK_SIZE under 64KB for AMD GPUs"

## API Endpoints

### POST /api/generate

Submit a new kernel generation request.

**Request Body:**
```json
{
  "ref_arch_src": "import torch\n\nclass Model(nn.Module):...",
  "gpu_arch": ["CDNA"],
  "backend": "triton",
  "model_name": "claude-sonnet-4.5",
  "server_type": "anthropic",
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
  "generated_kernel": "// Generated Triton kernel code...",
  "eval_result": "Evaluation results...",
  "error_message": null
}
```

### GET /api/requests

Get all recent generation requests.

### Status Values

- **pending**: Request submitted, waiting to start
- **processing**: Currently generating kernel
- **completed**: Generation finished successfully
- **failed**: Generation failed with error

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
    ├── index.html      # Frontend HTML
    └── guide.html      # User guide
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

To reset the database:
```bash
rm kernelbench_api.db
python -c "from api.database import db; db.init_database()"
```

## Production Deployment

For production deployment, consider:

1. **Use PostgreSQL** instead of SQLite for better concurrency
2. **Add Authentication** for secure access
3. **Set up Redis** for request queueing
4. **Use Docker** for containerization
5. **Add Rate Limiting** to prevent abuse
6. **Set up Monitoring** for system health

---

**TritonAgent** — Powered by AMD | [llm.amd.com](https://llm.amd.com)
