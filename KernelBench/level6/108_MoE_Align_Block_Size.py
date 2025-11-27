import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Original PyTorch implementation: MoE Align Block Size
    Sorts and aligns expert assignments to block boundaries for efficient batched computation
    
    Input: topk_ids [num_tokens, topk] - expert assignments per token
    Output: 
        - sorted_token_ids: token indices sorted by expert assignment
        - expert_ids: expert id for each aligned block
        - num_tokens_post_pad: total tokens after padding to block boundaries
    
    Based on sglang's moe_align_block_size operator from:
    sgl-kernel/csrc/moe/moe_align_kernel.cu
    """
    def __init__(self, num_experts, block_size):
        super(Model, self).__init__()
        self.num_experts = num_experts
        self.block_size = block_size

    def forward(self, topk_ids):
        """
        Args:
            topk_ids: [num_tokens * topk] flattened expert assignments
        
        Returns:
            sorted_token_ids: [num_tokens_post_pad] token indices sorted by expert
            expert_ids: [num_blocks] expert id for each block
            num_tokens_post_pad: scalar, total tokens after block alignment
        """
        numel = topk_ids.numel()
        device = topk_ids.device
        
        # Count tokens per expert
        expert_counts = torch.zeros(self.num_experts, dtype=torch.int32, device=device)
        for i in range(numel):
            expert_id = topk_ids[i].item()
            expert_counts[expert_id] += 1
        
        # Compute cumulative sum and pad to block boundaries
        cumsum = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=device)
        for e in range(self.num_experts):
            # Pad each expert's token count to block_size
            padded_count = ((expert_counts[e] + self.block_size - 1) // self.block_size) * self.block_size
            cumsum[e + 1] = cumsum[e] + padded_count
        
        num_tokens_post_pad = cumsum[-1].item()
        num_blocks = num_tokens_post_pad // self.block_size
        
        # Initialize outputs
        sorted_token_ids = torch.full((num_tokens_post_pad,), numel, dtype=torch.int32, device=device)
        expert_ids = torch.zeros(num_blocks, dtype=torch.int32, device=device)
        
        # Sort tokens by expert assignment
        expert_offsets = cumsum.clone()
        for token_id in range(numel):
            expert_id = topk_ids[token_id].item()
            pos = expert_offsets[expert_id].item()
            sorted_token_ids[pos] = token_id
            expert_offsets[expert_id] += 1
        
        # Assign expert IDs to blocks
        for e in range(self.num_experts):
            start_block = cumsum[e].item() // self.block_size
            end_block = cumsum[e + 1].item() // self.block_size
            for block_idx in range(start_block, end_block):
                expert_ids[block_idx] = e
        
        return sorted_token_ids, expert_ids, torch.tensor(num_tokens_post_pad, dtype=torch.int32, device=device)

num_tokens = 4096
topk = 2
num_experts = 64
block_size = 128

def get_inputs():
    """Generate topk_ids tensor of shape [num_tokens * topk]"""
    # Random expert assignments
    return [torch.randint(0, num_experts, (num_tokens * topk,), dtype=torch.int32)]

def get_init_inputs():
    return [num_experts, block_size]
