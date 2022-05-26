import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, reduction='none'):
        super().__init__()

        assert reduction in ['none', 'sum', 'mean']

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_logits: torch.Tensor, onehot_targets: torch.Tensor):
        ce_loss = torch.nn.functional.cross_entropy(input_logits, onehot_targets, reduction='none')
        input_probs_for_target = torch.exp(-ce_loss)
        loss = (1 - input_probs_for_target) ** self.gamma * ce_loss

        if self.reduction == 'sum':
            loss = loss.sum(dim=-1)
        elif self.reduction == 'mean':
            loss = loss.mean(dim=-1)

        return loss
