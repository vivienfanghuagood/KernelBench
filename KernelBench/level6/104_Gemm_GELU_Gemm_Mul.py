import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Original PyTorch implementation using separate Linear, GELU, and multiplication.
    Input shape: [batch_size, in_features]
    Output shape: [batch_size, out_features]
    
    This performs: output = GELU(x @ W1) * (x @ W2)
    where W1, W2 are [in_features, out_features]
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features, bias=False)
        self.fc2 = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        # x: [batch_size, in_features]
        gate = self.fc1(x)  # [batch_size, out_features]
        up = self.fc2(x)    # [batch_size, out_features]
        return F.gelu(gate) * up

batch_size = 1024
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features, dtype=torch.float16)]

def get_init_inputs():
    return [in_features, out_features]
