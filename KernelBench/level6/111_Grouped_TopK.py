import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Original PyTorch implementation: Grouped TopK for MoE routing
    Divides experts into groups and selects top-k experts within each group
    
    Input: gating_output [num_tokens, num_experts]
    Output: topk_weights [num_tokens, topk], topk_indices [num_tokens, topk]
    
    Based on grouped_topk functionality in sglang from:
    sgl-kernel/csrc/cpu/topk.cpp (grouped_topk_kernel_impl)
    
    This enables hierarchical expert selection where experts are organized into groups
    """
    def __init__(self, topk, num_expert_group, topk_group, renormalize=True):
        super(Model, self).__init__()
        self.topk = topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.renormalize = renormalize

    def forward(self, gating_output):
        """
        Args:
            gating_output: [num_tokens, num_experts] router logits
        
        Returns:
            topk_weights: [num_tokens, topk] normalized weights for selected experts
            topk_indices: [num_tokens, topk] global indices of selected experts
        """
        num_tokens, num_experts = gating_output.shape
        experts_per_group = num_experts // self.num_expert_group
        
        # Reshape to separate expert groups
        # [num_tokens, num_expert_group, experts_per_group]
        gating_grouped = gating_output.view(num_tokens, self.num_expert_group, experts_per_group)
        
        # Compute softmax within each group
        group_probs = F.softmax(gating_grouped, dim=-1, dtype=torch.float32)
        
        # Select top-k groups
        group_scores = group_probs.sum(dim=-1)  # [num_tokens, num_expert_group]
        topk_group_weights, topk_group_indices = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )
        
        # For each selected group, pick topk_per_group experts
        topk_per_group = self.topk // self.topk_group
        
        all_weights = []
        all_indices = []
        
        for i in range(num_tokens):
            token_weights = []
            token_indices = []
            
            for g_idx in topk_group_indices[i]:
                group_id = g_idx.item()
                group_expert_probs = group_probs[i, group_id]  # [experts_per_group]
                
                # Select top experts within this group
                expert_weights, expert_local_indices = torch.topk(
                    group_expert_probs, k=topk_per_group, dim=-1, sorted=False
                )
                
                # Convert to global expert indices
                expert_global_indices = expert_local_indices + group_id * experts_per_group
                
                token_weights.append(expert_weights)
                token_indices.append(expert_global_indices)
            
            all_weights.append(torch.cat(token_weights))
            all_indices.append(torch.cat(token_indices))
        
        topk_weights = torch.stack(all_weights)  # [num_tokens, topk]
        topk_indices = torch.stack(all_indices)  # [num_tokens, topk]
        
        # Renormalize if requested
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        return topk_weights, topk_indices

num_tokens = 4096
num_experts = 64
num_expert_group = 8
topk = 8
topk_group = 4

def get_inputs():
    """Generate gating output tensor of shape [num_tokens, num_experts]"""
    return [torch.randn(num_tokens, num_experts, dtype=torch.float32)]

def get_init_inputs():
    return [topk, num_expert_group, topk_group, True]
