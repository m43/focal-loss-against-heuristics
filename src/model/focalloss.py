import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (1 - pt) ** self.gamma * ce_loss
