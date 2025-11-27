"""
Batch Prefill Attention with Grouped Query Attention (GQA)

This kernel implements batch prefill attention with:
- Grouped Query Attention (GQA): Multiple query heads share the same KV heads
- Variable-length sequences using indptr arrays
- Causal masking for autoregressive generation
- KV cache indexing for efficient memory access

This is the core attention mechanism used in modern LLMs like Qwen3, Llama3, etc.

Reference: test_attention.py pytorch_attention implementation
Source: Inspired by SGLang's extend_attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Model(nn.Module):
    """
    Batch Prefill Attention with GQA and causal masking.
    
    Implements multi-head attention where multiple query heads can share
    the same key-value heads (Grouped Query Attention), with support for
    variable-length sequences in a batch.
    
    Args:
        num_q_heads: Number of query heads
        num_kv_heads: Number of key-value heads (num_q_heads % num_kv_heads == 0)
        head_dim: Dimension of each head
        v_head_dim: Dimension of value head (typically same as head_dim)
    """
    
    def __init__(self, num_q_heads: int, num_kv_heads: int, head_dim: int, v_head_dim: int):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"
        
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.num_groups = num_q_heads // num_kv_heads
    
    def forward(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        """
        Compute batch prefill attention with GQA and causal masking.
        
        Args:
            q: Query tensor [total_q_tokens, num_q_heads, head_dim]
            k_cache: Key cache [total_kv_blocks, num_kv_heads, head_dim]
            v_cache: Value cache [total_kv_blocks, num_kv_heads, v_head_dim]
            qo_indptr: Query output indptr [batch_size + 1], indices into q
            kv_indptr: KV indptr [batch_size + 1], indices into kv_indices
            kv_indices: KV block indices [total_kv_refs], indices into k_cache/v_cache
            softmax_scale: Scale factor for attention scores (typically 1/sqrt(head_dim))
        
        Returns:
            output: Attention output [total_q_tokens, num_q_heads, v_head_dim]
        """
        batch_size = len(qo_indptr) - 1
        output = torch.zeros_like(q) if q.shape[-1] == self.v_head_dim else \
                 torch.zeros(q.shape[0], self.num_q_heads, self.v_head_dim, 
                           dtype=q.dtype, device=q.device)
        
        for batch_idx in range(batch_size):
            # Get sequence ranges for this batch
            q_start = qo_indptr[batch_idx].item()
            q_end = qo_indptr[batch_idx + 1].item()
            kv_start = kv_indptr[batch_idx].item()
            kv_end = kv_indptr[batch_idx + 1].item()
            
            if q_end - q_start == 0 or kv_end - kv_start == 0:
                continue
            
            # Extract query sequence
            q_seq = q[q_start:q_end]  # [seq_len_q, num_q_heads, head_dim]
            
            # Extract key-value sequences using block indices
            block_indices = kv_indices[kv_start:kv_end]
            k_seq = k_cache[block_indices]  # [seq_len_kv, num_kv_heads, head_dim]
            v_seq = v_cache[block_indices]  # [seq_len_kv, num_kv_heads, v_head_dim]
            
            seq_len_q = q_end - q_start
            seq_len_kv = kv_end - kv_start
            
            # Process each query group
            for group_idx in range(self.num_groups):
                # Select query heads for this group
                q_head_start = group_idx * self.num_kv_heads
                q_head_end = (group_idx + 1) * self.num_kv_heads
                q_group = q_seq[:, q_head_start:q_head_end, :]
                # [seq_len_q, num_kv_heads, head_dim]
                
                # Compute attention scores: Q @ K^T
                # q_group: [seq_len_q, num_kv_heads, head_dim]
                # k_seq: [seq_len_kv, num_kv_heads, head_dim]
                # scores: [seq_len_q, num_kv_heads, seq_len_kv]
                scores = torch.einsum('qhd,khd->qhk', q_group, k_seq) * softmax_scale
                
                # Apply causal mask
                # Only allow attention to positions <= current position
                # Upper triangular mask with appropriate diagonal offset
                mask = torch.triu(
                    torch.ones(seq_len_q, seq_len_kv, dtype=torch.bool, device=q.device),
                    diagonal=seq_len_kv - seq_len_q + 1
                )
                scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))
                
                # Softmax over key positions
                attn_weights = F.softmax(scores, dim=-1)
                # [seq_len_q, num_kv_heads, seq_len_kv]
                
                # Multiply by values: attn_weights @ V
                # attn_weights: [seq_len_q, num_kv_heads, seq_len_kv]
                # v_seq: [seq_len_kv, num_kv_heads, v_head_dim]
                # o_group: [seq_len_q, num_kv_heads, v_head_dim]
                o_group = torch.einsum('qhk,khd->qhd', attn_weights, v_seq)
                
                # Write back to output
                output[q_start:q_end, q_head_start:q_head_end, :] = o_group
        
        return output


def get_inputs():
    """
    Generate inputs for Qwen3-Next style attention:
    - Batch size: 2
    - Sequence lengths: variable (2048, 2048 for queries; 4096, 4096 for KV)
    - 16 query heads, 2 KV heads (8x GQA)
    - Head dimension: 256
    """
    batch_size = 2
    seq_lens_q = [2048, 2048]
    seq_lens_kv = [4096, 4096]
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 256
    v_head_dim = 256
    
    device = 'cuda'
    dtype = torch.float16
    
    total_q_tokens = sum(seq_lens_q)
    total_kv_tokens = sum(seq_lens_kv)
    
    # Query: [total_q_tokens, num_q_heads, head_dim]
    q = torch.randn(total_q_tokens, num_q_heads, head_dim, dtype=dtype, device=device) * 0.1
    
    # KV cache: [total_kv_tokens, num_kv_heads, head_dim/v_head_dim]
    k_cache = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype, device=device) * 0.1
    v_cache = torch.randn(total_kv_tokens, num_kv_heads, v_head_dim, dtype=dtype, device=device) * 0.1
    
    # Indptr arrays for variable-length sequences
    qo_indptr = torch.tensor([0] + [sum(seq_lens_q[:i+1]) for i in range(batch_size)],
                              dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0] + [sum(seq_lens_kv[:i+1]) for i in range(batch_size)],
                              dtype=torch.int32, device=device)
    
    # KV indices: simple sequential mapping (one block per token)
    kv_indices = torch.arange(total_kv_tokens, dtype=torch.int32, device=device)
    
    # Softmax scale: 1/sqrt(head_dim)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, softmax_scale]


def get_init_inputs():
    """
    Initialize model with Qwen3-Next configuration:
    - 16 query heads
    - 2 KV heads (8x grouped query attention)
    - 256 head dimension
    """
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 256
    v_head_dim = 256
    
    return [num_q_heads, num_kv_heads, head_dim, v_head_dim]
