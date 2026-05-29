from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modeling import TemporalSegmentNet, pairwise_rank_loss


class ModelingTest(unittest.TestCase):
    def test_tsn_forward_backward(self) -> None:
        torch.manual_seed(7)
        model = TemporalSegmentNet(input_dim=12, hidden_dim=16, num_segments=4, dropout=0.1)
        x = torch.randn(8, 20, 12)
        y = torch.randn(8)
        trade_date = torch.tensor([20260105] * 4 + [20260106] * 4)

        pred = model(x)
        self.assertEqual(tuple(pred.shape), (8,))
        loss = torch.nn.functional.mse_loss(pred, y) + 0.2 * pairwise_rank_loss(
            pred,
            y,
            trade_date,
        )
        loss.backward()
        grad_norm = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()

