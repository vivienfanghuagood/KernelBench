import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Original PyTorch implementation: GELU(input[:, :d]) * input[:, d:]
    Input shape: [batch_size, 2 * out_features]
    Output shape: [batch_size, out_features]
    """
    def __init__(self, out_features):
        super(Model, self).__init__()
        self.out_features = out_features

    def forward(self, x):
        # x: [batch_size, 2 * out_features]
        d = self.out_features
        gate = x[:, :d]      # [batch_size, out_features]
        up = x[:, d:]        # [batch_size, out_features]
        return F.gelu(gate) * up

batch_size = 64 * 1024
out_features = 8192

def get_inputs():
    """Generate input tensor of shape [batch_size, 2 * out_features]"""
    return [torch.rand(batch_size, 2 * out_features, dtype=torch.float16)]

def get_init_inputs():
    return [out_features]
