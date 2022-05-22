import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def forward(self, input_logits: torch.Tensor, onehot_targets: torch.Tensor):
        input_probs = torch.softmax(input_logits, dim=-1)
        ce_loss = torch.nn.functional.cross_entropy(input_probs, onehot_targets, reduction='none')
        input_probs_for_target = torch.exp(-ce_loss)
        return (1 - input_probs_for_target) ** self.gamma * ce_loss
