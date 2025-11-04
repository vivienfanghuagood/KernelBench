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
            'openai': 'gpt-4o-2024-08-06',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'google': 'gemini-1.5-flash-002'
        };
        modelNameInput.value = modelMap[serverType] || 'deepseek-coder';
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
        } finally {
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
            selectedArches.push('Ada'); // Default
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
        const generatedKernel = document.getElementById('generatedKernel');
        const evalResults = document.getElementById('evalResults');

        generatedKernel.textContent = status.generated_kernel || 'No kernel generated';
        evalResults.textContent = status.eval_result || 'No evaluation results';

        resultsSection.style.display = 'block';
        this.hideProgress();

        // Re-highlight code
        if (window.Prism) {
            Prism.highlightAll();
        }
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
            
            row.innerHTML = `
                <td><code>${request.id.substring(0, 8)}...</code></td>
                <td><span class="badge bg-${statusClass}">${request.status}</span></td>
                <td>${request.backend}</td>
                <td>${request.model_name}</td>
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
        
        if (show) {
            spinner.style.display = 'inline-block';
            button.disabled = true;
            button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Generating...';
        } else {
            spinner.style.display = 'none';
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

    showResults() {
        document.getElementById('resultsSection').style.display = 'block';
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
        refArchSrc.value = `import torch

def square_matrix_multiplication(A, B):
    """
    Simple square matrix multiplication
    A: [N, N] 
    B: [N, N]
    Returns: C = A @ B [N, N]
    """
    return torch.matmul(A, B)`;
    };
    
    refArchSrc.parentNode.appendChild(sampleButton);
});