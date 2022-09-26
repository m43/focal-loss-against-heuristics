import unittest

import torch
import torch.nn.functional as F

from src.model.focalloss import FocalLoss
from src.infersent.src.losses import FocalLoss as FocalLossInferSent


class TestFocalLoss(unittest.TestCase):
    def test_gamma_zero_same_as_CE(self):
        """
        Test that when gamma is 0 we are similar to CE
        """
        BATCH_SIZE = 32
        NUM_CLASSES = 5

        logits = torch.randn((BATCH_SIZE, NUM_CLASSES)) * 5
        targets = F.one_hot(
            torch.randint(high=NUM_CLASSES, size=(BATCH_SIZE,)),
            num_classes=NUM_CLASSES
        ).float()

        expected = F.cross_entropy(logits, targets, reduction='none')
        actual = FocalLoss(gamma=0., reduction='none').forward(logits, targets)

        self.assertTrue(
            torch.all(torch.isclose(expected, actual)),
        )

    def test_formula_applied_correctly(self):
        """
        Test that focal loss computes the loss as we would expect it for a few samples.
        """
        logits = torch.tensor([
            [10, 9, -5],
            [0, 2, -10]
        ]).float()
        probs = F.softmax(logits, dim=-1).max(dim=-1)[0]

        targets = torch.tensor([
            [1, 0, 0],
            [0, 1, 0]
        ]).float()

        for gamma in [1., 2., 5., 10.]:
            expected = - (((1 - probs) ** gamma) * torch.log(probs))
            actual = FocalLoss(gamma=gamma, reduction='none').forward(logits, targets)

            self.assertTrue(
                torch.all(torch.isclose(expected, actual)),
            )

    def test_formula_applied_correctly_infersent(self):
        """
        Test that focal loss computes the loss as we would expect it for a few samples.
        """
        logits = torch.tensor([
            [10, 9, -5],
            [0, 2, -10]
        ]).float()
        probs = F.softmax(logits, dim=-1).max(dim=-1)[0]

        targets = torch.tensor([
            0,
            1
        ]).long()

        for gamma in [1., 2., 5., 10.]:
            expected = - (((1 - probs) ** gamma) * torch.log(probs))
            actual = torch.tensor([FocalLossInferSent(gamma=gamma).forward
                                   (logits[i].unsqueeze(0), targets[i].unsqueeze(0)) for i in range(2)])

            self.assertTrue(
                torch.all(torch.isclose(expected, actual)),
            )


if __name__ == '__main__':
    unittest.main()
