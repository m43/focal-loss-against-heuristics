import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, gamma: float, reduction: str):
        """
            parameter  gamma: Exponent of the modulating factor (1 - p_t) to
                   balance easy vs hard examples.
            parameter reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.

        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                inputs: torch.Tensor,
                targets: torch.Tensor):
        """
        Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
                    In logits
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor with the reduction option applied.
        """
        p = torch.softmax(inputs, dim=-1)

        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = torch.gather(p, dim=-1, index=targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss
