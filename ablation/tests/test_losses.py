import unittest

try:
    import torch

    from ablation.losses import (
        ExponentialLogarithmicLoss,
        hard_transition_probability_matrix,
        vtpm_alignment_loss,
    )
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "当前环境没有安装PyTorch")
class AblationLossTest(unittest.TestCase):
    def test_matching_vtpm_has_near_zero_loss_and_gradient(self):
        target = torch.tensor([[[0, 0], [1, 1], [2, 2]]])
        logits = torch.full((1, 3, 3, 2), -20.0)
        logits.scatter_(1, target.unsqueeze(1), 20.0)
        logits.requires_grad_(True)
        loss, _ = vtpm_alignment_loss(logits, target)
        self.assertLess(float(loss), 1e-8)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_hard_vtpm_rows_are_normalized(self):
        target = torch.tensor([[[0], [0], [1], [1], [2]]])
        matrix = hard_transition_probability_matrix(target, 3)
        nonempty = matrix.sum(dim=1) > 0
        torch.testing.assert_close(
            matrix.sum(dim=1)[nonempty], torch.ones(int(nonempty.sum()))
        )

    def test_ell_is_finite_and_differentiable(self):
        logits = torch.randn(2, 3, 4, 5, requires_grad=True)
        target = torch.randint(0, 3, (2, 4, 5))
        criterion = ExponentialLogarithmicLoss(torch.ones(3))
        loss, components = criterion(logits, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("ell_dice", components)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ell_near_perfect_prediction_has_finite_gradient(self):
        target = torch.tensor([[[0, 1, 2]]])
        logits = torch.full((1, 3, 1, 3), -100.0)
        logits.scatter_(1, target.unsqueeze(1), 100.0)
        logits.requires_grad_(True)
        criterion = ExponentialLogarithmicLoss(torch.ones(3))
        loss, _ = criterion(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "需要CUDA")
    def test_full_section_vtpm_is_finite_under_amp(self):
        logits = torch.randn(1, 6, 255, 701, device="cuda", requires_grad=True)
        target = torch.randint(0, 6, (1, 255, 701), device="cuda")
        with torch.autocast(device_type="cuda", enabled=True):
            loss, _ = vtpm_alignment_loss(logits, target, smoothing=1e-6)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
