import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Neural Sparse Attention (NSA) layer inspired by DeepSeek V3.2-Exp.
    
    This model implements a simplified version of the indexer-based sparse attention mechanism
    from DeepSeek V3.2-Exp, which computes top-k indices for efficient sparse attention.
    
    Key components:
    - Indexer: Computes sparse attention indices using query and key projections
    - Q/K projections: Transform input to query and key representations
    - Top-k selection: Selects most relevant positions for attention
    - Hadamard transform: Applies rotation to activations for better mixing
    """
    
    def __init__(self, dim, n_heads, head_dim, index_topk, q_lora_rank):
        super(Model, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        
        # Query projection using low-rank decomposition
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        
        # Key projection
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        
        # Key normalization (LayerNorm)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
        # Weights projection for combining multi-head scores
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        
        self.softmax_scale = self.head_dim ** -0.5
    
    def rotate_activation(self, x):
        """
        Applies a simple rotation transformation to activations.
        In the full implementation, this uses Hadamard transform.
        Here we use a simple permutation for demonstration.
        """
        # Simple rotation: split and swap halves
        hidden_size = x.size(-1)
        mid = hidden_size // 2
        x_left = x[..., :mid]
        x_right = x[..., mid:]
        # Swap and negate one half for rotation effect
        return torch.cat([-x_right, x_left], dim=-1) * (hidden_size ** -0.5)
    
    def forward(self, x, qr):
        """
        Forward pass of the NSA indexer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim)
            qr (torch.Tensor): Query representation from q_lora of shape (batch_size, seq_len, q_lora_rank)
        
        Returns:
            torch.Tensor: Top-k indices for sparse attention of shape (batch_size, seq_len, n_heads, index_topk)
        """
        bsz, seqlen, _ = x.size()
        
        # Project query from low-rank representation
        q = self.wq_b(qr)  # (bsz, seqlen, n_heads * head_dim)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)  # (bsz, seqlen, n_heads, head_dim)
        
        # Project and normalize key
        k = self.wk(x)  # (bsz, seqlen, head_dim)
        k = self.k_norm(k)  # (bsz, seqlen, head_dim)
        
        # Apply rotation transformation
        q = self.rotate_activation(q)  # (bsz, seqlen, n_heads, head_dim)
        k = self.rotate_activation(k.unsqueeze(2))  # (bsz, seqlen, 1, head_dim)
        
        # Compute attention scores: Q @ K^T
        # q: (bsz, seqlen, n_heads, head_dim)
        # k: (bsz, seqlen, 1, head_dim) -> broadcast to (bsz, seqlen, n_heads, head_dim)
        k = k.expand(-1, -1, self.n_heads, -1)
        
        # Compute scores for each head
        # For each position, compute similarity with all other positions
        scores = torch.zeros(bsz, seqlen, self.n_heads, seqlen, device=x.device, dtype=x.dtype)
        for i in range(seqlen):
            # q_i: (bsz, n_heads, head_dim)
            q_i = q[:, i, :, :]
            # k_all: (bsz, seqlen, n_heads, head_dim)
            # Compute dot product: (bsz, n_heads, head_dim) @ (bsz, n_heads, seqlen, head_dim)^T
            k_all = k.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
            scores_i = torch.einsum('bhd,bhsd->bhs', q_i, k_all)  # (bsz, n_heads, seqlen)
            scores[:, i, :, :] = scores_i
        
        # Apply softmax scale
        scores = scores * self.softmax_scale
        
        # Get importance weights from input
        weights = self.weights_proj(x.float())  # (bsz, seqlen, n_heads)
        weights = weights * (self.n_heads ** -0.5)
        
        # Combine with scores
        weights = weights.unsqueeze(-1)  # (bsz, seqlen, n_heads, 1)
        index_score = scores + weights  # (bsz, seqlen, n_heads, seqlen)
        
        # Apply causal mask (can only attend to past positions)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), device=x.device),
            diagonal=1
        )
        index_score = index_score + causal_mask.unsqueeze(0).unsqueeze(1)  # (bsz, seqlen, n_heads, seqlen)
        
        # Select top-k indices for each position and head
        topk_indices = index_score.topk(min(self.index_topk, seqlen), dim=-1)[1]  # (bsz, seqlen, n_heads, index_topk)
        
        return topk_indices


# Configuration
batch_size = 8
seq_len = 512
dim = 7168  # DeepSeek-V3.2-Exp hidden size
n_heads = 64  # Number of indexer heads
head_dim = 128  # Dimension per head
index_topk = 2048  # Number of positions to select (sparse attention)
q_lora_rank = 1536  # Low-rank dimension for query


def get_inputs():
    """
    Generate sample inputs for the NSA model.
    
    Returns:
        list: [x, qr] where:
            - x: input tensor (batch_size, seq_len, dim)
            - qr: query representation (batch_size, seq_len, q_lora_rank)
    """
    x = torch.randn(batch_size, seq_len, dim, dtype=torch.float32)
    qr = torch.randn(batch_size, seq_len, q_lora_rank, dtype=torch.float32)
    return [x, qr]


def get_init_inputs():
    """
    Get initialization parameters for the model.
    
    Returns:
        list: [dim, n_heads, head_dim, index_topk, q_lora_rank]
    """
    return [dim, n_heads, head_dim, index_topk, q_lora_rank]
