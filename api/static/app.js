class KernelBenchAPI {
    constructor() {
        this.baseURL = '';
        this.currentRequestId = null;
        this.pollingInterval = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadRecentRequests();
    }

    setupEventListeners() {
        const form = document.getElementById('kernelForm');
        form.addEventListener('submit', (e) => this.handleFormSubmit(e));

        // Update model name based on server type
        const serverTypeSelect = document.getElementById('serverType');
        serverTypeSelect.addEventListener('change', (e) => this.updateModelName(e.target.value));
    }

    updateModelName(serverType) {
        const modelNameInput = document.getElementById('modelName');
        const modelMap = {
            'deepseek': 'deepseek-coder',
            'openai': 'gpt-5',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'google': 'gemini-1.5-flash-002'
        };
        modelNameInput.value = modelMap[serverType] || 'gpt-5';
    }

    async handleFormSubmit(e) {
        e.preventDefault();
        
        const formData = this.getFormData();
        if (!formData) return;

        this.showLoading(true);
        this.hideError();
        this.hideResults();

        try {
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Failed to submit request');
            }

            this.currentRequestId = result.request_id;
            this.showProgress();
            this.startPolling();

        } catch (error) {
            this.showError('Failed to submit request: ' + error.message);
            this.showLoading(false);
        }
    }

    getFormData() {
        const refArchSrc = document.getElementById('refArchSrc').value.trim();
        if (!refArchSrc) {
            this.showError('Please provide reference architecture source code');
            return null;
        }

        const gpuArchSelect = document.getElementById('gpuArch');
        const selectedArches = Array.from(gpuArchSelect.selectedOptions).map(option => option.value);
        if (selectedArches.length === 0) {
            selectedArches.push('4090'); // Default
        }

        return {
            ref_arch_src: refArchSrc,
            gpu_arch: selectedArches,
            backend: document.getElementById('backend').value,
            model_name: document.getElementById('modelName').value,
            server_type: document.getElementById('serverType').value,
            max_tokens: parseInt(document.getElementById('maxTokens').value),
            temperature: parseFloat(document.getElementById('temperature').value)
        };
    }

    async startPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }

        this.pollingInterval = setInterval(async () => {
            try {
                const status = await this.checkStatus(this.currentRequestId);
                this.updateProgress(status);

                if (status.status === 'completed' || status.status === 'failed') {
                    this.stopPolling();
                    this.showLoading(false);
                    this.hideProgress();
                    
                    if (status.status === 'completed') {
                        this.showResults(status);
                    } else {
                        this.showError('Generation failed: ' + (status.error_message || 'Unknown error'));
                    }
                    this.loadRecentRequests(); // Refresh the table
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 2000); // Poll every 2 seconds
    }

    stopPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    async checkStatus(requestId) {
        const response = await fetch(`/api/status/${requestId}`);
        if (!response.ok) {
            throw new Error('Failed to check status');
        }
        return await response.json();
    }

    updateProgress(status) {
        const statusBadge = document.getElementById('statusBadge');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const requestIdSpan = document.getElementById('requestId');

        requestIdSpan.textContent = `(${status.request_id.substring(0, 8)}...)`;

        const statusMap = {
            'pending': { class: 'bg-secondary', progress: 25, text: 'Request queued...' },
            'processing': { class: 'bg-primary', progress: 50, text: 'Generating kernel...' },
            'completed': { class: 'bg-success', progress: 100, text: 'Generation completed!' },
            'failed': { class: 'bg-danger', progress: 100, text: 'Generation failed' }
        };

        const statusInfo = statusMap[status.status] || statusMap['pending'];
        
        statusBadge.className = `badge status-badge ${statusInfo.class}`;
        statusBadge.textContent = status.status.charAt(0).toUpperCase() + status.status.slice(1);
        
        progressBar.style.width = `${statusInfo.progress}%`;
        progressText.textContent = statusInfo.text;

        if (status.started_at) {
            const startTime = new Date(status.started_at);
            const elapsed = Math.floor((Date.now() - startTime.getTime()) / 1000);
            progressText.textContent += ` (${elapsed}s elapsed)`;
        }
    }

    showResults(status) {
        const resultsSection = document.getElementById('resultsSection');
        const refArchSrc = document.getElementById('refArchSrc');
        const generatedKernel = document.getElementById('generatedKernel');
        const evalResults = document.getElementById('evalResults');

        // Display reference architecture back in the input textarea
        refArchSrc.value = status.ref_arch_src || 'No reference code';
        
        // Display generated kernel
        generatedKernel.textContent = status.generated_kernel || 'No kernel generated';
        
        // Parse and display evaluation results
        evalResults.innerHTML = this.formatEvalResults(status.eval_result);

        resultsSection.style.display = 'block';

        // Re-highlight code
        if (window.Prism) {
            Prism.highlightAll();
        }
    }

    formatEvalResults(evalResultStr) {
        if (!evalResultStr) {
            return '<div class="alert alert-warning">No evaluation results</div>';
        }

        try {
            // Parse the eval result string
            // Expected format: "compiled=True correctness=True metadata={'hardware': 'NVIDIA GeForce RTX 4090', ...} runtime=1.97 runtime_stats={...}"
            const result = this.parseEvalString(evalResultStr);
            
            let html = '<div class="eval-results-formatted">';
            
            // Compilation status
            const compiledBadge = result.compiled 
                ? '<span class="badge bg-success">✓ Compiled</span>'
                : '<span class="badge bg-danger">✗ Failed to Compile</span>';
            
            // Correctness status
            const correctnessBadge = result.correctness 
                ? '<span class="badge bg-success">✓ Correct</span>'
                : '<span class="badge bg-danger">✗ Incorrect</span>';
            
            html += `
                <div class="mb-3">
                    <strong>Status:</strong> ${compiledBadge} ${correctnessBadge}
                </div>
            `;
            
            // Hardware info
            if (result.metadata && result.metadata.hardware) {
                html += `
                    <div class="mb-3">
                        <strong>Hardware:</strong> ${result.metadata.hardware}
                        ${result.metadata.device ? ` (Device ${result.metadata.device})` : ''}
                    </div>
                `;
            }
            
            // Correctness trials
            if (result.metadata && result.metadata.correctness_trials) {
                html += `
                    <div class="mb-3">
                        <strong>Correctness Trials:</strong> ${result.metadata.correctness_trials}
                    </div>
                `;
            }
            
            // Speedup (if available)
            if (result.speedup !== null && result.speedup > 0) {
                const speedupClass = result.speedup > 1 ? 'bg-success' : result.speedup < 1 ? 'bg-warning' : 'bg-secondary';
                html += `
                    <div class="mb-3">
                        <strong>Speedup:</strong> <span class="badge ${speedupClass} fs-5">${result.speedup.toFixed(2)}x</span>
                        ${result.speedup > 1 ? '🚀 Faster than reference!' : result.speedup < 1 ? '⚠️ Slower than reference' : ''}
                    </div>
                `;
            }
            
            // Performance comparison table
            if (result.ref_runtime !== null && result.runtime !== null) {
                html += `
                    <div class="mb-3">
                        <strong>Performance Comparison:</strong>
                        <table class="table table-sm table-bordered mt-2">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Mean Runtime</th>
                                    <th>Std Dev</th>
                                    <th>Min</th>
                                    <th>Max</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr class="table-light">
                                    <td><strong>Reference (PyTorch)</strong></td>
                                    <td>${result.ref_runtime.toFixed(2)} ms</td>
                                    <td>${result.ref_runtime_stats.std ? result.ref_runtime_stats.std.toFixed(4) : 'N/A'} ms</td>
                                    <td>${result.ref_runtime_stats.min ? result.ref_runtime_stats.min.toFixed(2) : 'N/A'} ms</td>
                                    <td>${result.ref_runtime_stats.max ? result.ref_runtime_stats.max.toFixed(2) : 'N/A'} ms</td>
                                </tr>
                                <tr class="table-primary">
                                    <td><strong>Custom Kernel</strong></td>
                                    <td>${result.runtime.toFixed(2)} ms</td>
                                    <td>${result.runtime_stats.std ? result.runtime_stats.std.toFixed(4) : 'N/A'} ms</td>
                                    <td>${result.runtime_stats.min ? result.runtime_stats.min.toFixed(2) : 'N/A'} ms</td>
                                    <td>${result.runtime_stats.max ? result.runtime_stats.max.toFixed(2) : 'N/A'} ms</td>
                                </tr>
                            </tbody>
                        </table>
                        <small class="text-muted">Number of trials: ${result.runtime_stats.num_trials || 'N/A'}</small>
                    </div>
                `;
            } else {
                // Show only custom runtime if ref_runtime is not available (legacy support)
                if (result.runtime !== null) {
                    html += `
                        <div class="mb-3">
                            <strong>Runtime:</strong> <span class="badge bg-info">${result.runtime.toFixed(2)} ms</span>
                        </div>
                    `;
                }
                
                // Runtime statistics
                if (result.runtime_stats && Object.keys(result.runtime_stats).length > 0) {
                    const stats = result.runtime_stats;
                    html += `
                        <div class="mb-3">
                            <strong>Runtime Statistics:</strong>
                            <table class="table table-sm table-bordered mt-2">
                                <thead>
                                    <tr>
                                        <th>Mean</th>
                                        <th>Std Dev</th>
                                        <th>Min</th>
                                        <th>Max</th>
                                        <th>Trials</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>${stats.mean ? stats.mean.toFixed(2) : 'N/A'} ms</td>
                                        <td>${stats.std ? stats.std.toFixed(4) : 'N/A'} ms</td>
                                        <td>${stats.min ? stats.min.toFixed(2) : 'N/A'} ms</td>
                                        <td>${stats.max ? stats.max.toFixed(2) : 'N/A'} ms</td>
                                        <td>${stats.num_trials || 'N/A'}</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    `;
                }
            }
            
            html += '</div>';
            
            return html;
            
        } catch (error) {
            console.error('Error formatting eval results:', error);
            // Fallback to showing raw text
            return `<pre class="text-muted">${evalResultStr}</pre>`;
        }
    }

    parseEvalString(str) {
        const result = {
            compiled: null,
            correctness: null,
            metadata: {},
            runtime: null,
            runtime_stats: {},
            ref_runtime: null,
            ref_runtime_stats: {},
            speedup: null
        };

        try {
            // Extract compiled
            const compiledMatch = str.match(/compiled=(True|False)/);
            if (compiledMatch) {
                result.compiled = compiledMatch[1] === 'True';
            }

            // Extract correctness
            const correctnessMatch = str.match(/correctness=(True|False)/);
            if (correctnessMatch) {
                result.correctness = correctnessMatch[1] === 'True';
            }

            // Extract metadata - need to handle nested braces
            const metadataMatch = str.match(/metadata=(\{[^}]*\})/);
            if (metadataMatch) {
                try {
                    const metadataStr = metadataMatch[1]
                        .replace(/'/g, '"')
                        .replace(/\((\d+)\s*\/\s*(\d+)\)/g, '"($1 / $2)"'); // Handle fractions like (3 / 3)
                    result.metadata = JSON.parse(metadataStr);
                } catch (e) {
                    console.error('Error parsing metadata:', e);
                }
            }

            // Extract runtime (but not runtime_stats, ref_runtime, or ref_runtime_stats)
            const runtimeMatch = str.match(/\bruntime=([\d.]+)(?=\s|$|runtime_stats|ref_)/);
            if (runtimeMatch) {
                result.runtime = parseFloat(runtimeMatch[1]);
            }

            // Extract ref_runtime
            const refRuntimeMatch = str.match(/ref_runtime=([\d.]+)(?=\s|$|ref_runtime_stats)/);
            if (refRuntimeMatch) {
                result.ref_runtime = parseFloat(refRuntimeMatch[1]);
            }

            // Extract speedup
            const speedupMatch = str.match(/speedup=([\d.]+)/);
            if (speedupMatch) {
                result.speedup = parseFloat(speedupMatch[1]);
            }

            // Extract runtime_stats - need to handle nested dictionary
            // Use a more robust method to extract the dictionary
            const statsStartIndex = str.indexOf('runtime_stats={');
            if (statsStartIndex !== -1) {
                const statsStart = statsStartIndex + 'runtime_stats='.length;
                let braceCount = 0;
                let statsEnd = statsStart;
                
                // Find the matching closing brace
                for (let i = statsStart; i < str.length; i++) {
                    if (str[i] === '{') braceCount++;
                    if (str[i] === '}') {
                        braceCount--;
                        if (braceCount === 0) {
                            statsEnd = i + 1;
                            break;
                        }
                    }
                }
                
                if (statsEnd > statsStart) {
                    const statsStr = str.substring(statsStart, statsEnd)
                        .replace(/'/g, '"')
                        .replace(/\bNone\b/g, 'null')
                        .replace(/\bTrue\b/g, 'true')
                        .replace(/\bFalse\b/g, 'false');
                    
                    try {
                        result.runtime_stats = JSON.parse(statsStr);
                    } catch (e) {
                        console.error('Error parsing runtime_stats:', e, 'String was:', statsStr);
                    }
                }
            }

            // Extract ref_runtime_stats
            const refStatsStartIndex = str.indexOf('ref_runtime_stats={');
            if (refStatsStartIndex !== -1) {
                const refStatsStart = refStatsStartIndex + 'ref_runtime_stats='.length;
                let braceCount = 0;
                let refStatsEnd = refStatsStart;
                
                // Find the matching closing brace
                for (let i = refStatsStart; i < str.length; i++) {
                    if (str[i] === '{') braceCount++;
                    if (str[i] === '}') {
                        braceCount--;
                        if (braceCount === 0) {
                            refStatsEnd = i + 1;
                            break;
                        }
                    }
                }
                
                if (refStatsEnd > refStatsStart) {
                    const refStatsStr = str.substring(refStatsStart, refStatsEnd)
                        .replace(/'/g, '"')
                        .replace(/\bNone\b/g, 'null')
                        .replace(/\bTrue\b/g, 'true')
                        .replace(/\bFalse\b/g, 'false');
                    
                    try {
                        result.ref_runtime_stats = JSON.parse(refStatsStr);
                    } catch (e) {
                        console.error('Error parsing ref_runtime_stats:', e, 'String was:', refStatsStr);
                    }
                }
            }

        } catch (error) {
            console.error('Error parsing eval string:', error);
        }

        return result;
    }

    async loadRecentRequests() {
        try {
            const response = await fetch('/api/requests?limit=10');
            const data = await response.json();
            this.updateRequestsTable(data.requests || []);
        } catch (error) {
            console.error('Failed to load recent requests:', error);
        }
    }

    updateRequestsTable(requests) {
        const tbody = document.querySelector('#requestsTable tbody');
        tbody.innerHTML = '';

        requests.forEach(request => {
            const row = document.createElement('tr');
            
            const statusClass = {
                'pending': 'secondary',
                'processing': 'primary',
                'completed': 'success',
                'failed': 'danger'
            }[request.status] || 'secondary';

            const createdAt = new Date(request.created_at).toLocaleString();
            
            // Parse eval_result to extract compiled, correctness, runtime, and speedup
            let compiled = '-';
            let correctness = '-';
            let runtime = '-';
            let speedup = '-';
            
            if (request.eval_result && request.status === 'completed') {
                try {
                    const evalData = this.parseEvalString(request.eval_result);
                    
                    if (evalData.compiled !== null) {
                        compiled = evalData.compiled 
                            ? '<span class="badge bg-success">✓</span>' 
                            : '<span class="badge bg-danger">✗</span>';
                    }
                    
                    if (evalData.correctness !== null) {
                        correctness = evalData.correctness 
                            ? '<span class="badge bg-success">✓</span>' 
                            : '<span class="badge bg-danger">✗</span>';
                    }
                    
                    if (evalData.runtime !== null) {
                        runtime = evalData.runtime.toFixed(2);
                    }
                    
                    if (evalData.speedup !== null && evalData.speedup > 0) {
                        const speedupClass = evalData.speedup > 1 ? 'text-success fw-bold' : evalData.speedup < 1 ? 'text-warning' : '';
                        speedup = `<span class="${speedupClass}">${evalData.speedup.toFixed(2)}x</span>`;
                    }
                } catch (e) {
                    console.error('Error parsing eval_result for table:', e);
                }
            }
            
            row.innerHTML = `
                <td><code>${request.id.substring(0, 8)}...</code></td>
                <td><span class="badge bg-${statusClass}">${request.status}</span></td>
                <td>${request.backend}</td>
                <td>${request.model_name}</td>
                <td>${compiled}</td>
                <td>${correctness}</td>
                <td>${runtime}</td>
                <td>${speedup}</td>
                <td>${createdAt}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="api.viewRequest('${request.id}')">
                        View
                    </button>
                </td>
            `;
            
            tbody.appendChild(row);
        });
    }

    async viewRequest(requestId) {
        try {
            const status = await this.checkStatus(requestId);
            
            if (status.status === 'completed') {
                this.showResults(status);
                this.hideProgress();
            } else {
                // Show current status
                this.currentRequestId = requestId;
                this.updateProgress(status);
                this.showProgress();
                
                if (status.status === 'processing') {
                    this.startPolling();
                }
            }
            
            this.hideError();
        } catch (error) {
            this.showError('Failed to load request: ' + error.message);
        }
    }

    showLoading(show) {
        const spinner = document.querySelector('.loading-spinner');
        const button = document.querySelector('button[type="submit"]');
        
        if (!button) return; // Guard against missing button
        
        if (show) {
            if (spinner) spinner.style.display = 'inline-block';
            button.disabled = true;
            button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';
        } else {
            if (spinner) spinner.style.display = 'none';
            button.disabled = false;
            button.innerHTML = 'Generate Kernel';
        }
    }

    showProgress() {
        document.querySelector('.progress-section').style.display = 'block';
    }

    hideProgress() {
        document.querySelector('.progress-section').style.display = 'none';
    }

    hideResults() {
        document.getElementById('resultsSection').style.display = 'none';
    }

    showError(message) {
        const errorAlert = document.getElementById('errorAlert');
        errorAlert.textContent = message;
        errorAlert.style.display = 'block';
    }

    hideError() {
        document.getElementById('errorAlert').style.display = 'none';
    }
}

// Initialize the API client
const api = new KernelBenchAPI();

// Add sample data button (for testing)
document.addEventListener('DOMContentLoaded', function() {
    const refArchSrc = document.getElementById('refArchSrc');
    
    // Add a button to load sample data
    const sampleButton = document.createElement('button');
    sampleButton.type = 'button';
    sampleButton.className = 'btn btn-sm btn-outline-secondary mt-2';
    sampleButton.textContent = 'Load Sample Code';
    sampleButton.onclick = () => {
        refArchSrc.value = `
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed`;
    };
    
    refArchSrc.parentNode.appendChild(sampleButton);
});