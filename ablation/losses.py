"""论文定义的ELL损失和可微VTPM层序约束。"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialLogarithmicLoss(nn.Module):
    """指数对数损失：指数化Dice损失与类别加权交叉熵的组合。"""

    def __init__(
        self,
        class_weights: torch.Tensor,
        dice_weight: float = 0.8,
        cross_entropy_weight: float = 0.2,
        dice_gamma: float = 0.3,
        cross_entropy_gamma: float = 0.3,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if class_weights.ndim != 1:
            raise ValueError("class_weights必须是一维张量")
        self.register_buffer("class_weights", class_weights.float())
        self.dice_weight = dice_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_gamma = dice_gamma
        self.cross_entropy_gamma = cross_entropy_gamma
        self.epsilon = epsilon

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if logits.ndim != 4 or target.ndim != 3:
            raise ValueError("logits和target应分别为[B,C,H,W]和[B,H,W]")
        probabilities = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=probabilities.dtype)

        reduce_axes = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dim=reduce_axes)
        denominator = torch.sum(probabilities + one_hot, dim=reduce_axes)
        dice = (2.0 * intersection + self.epsilon) / (
            denominator + self.epsilon
        )
        # gamma小于1时，x^gamma在x=0处导数发散；避开概率精确等于1的端点。
        dice = dice.clamp(min=self.epsilon, max=1.0 - self.epsilon)
        dice_term = torch.pow(-torch.log(dice), self.dice_gamma).mean()

        true_probability = probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
        true_probability = true_probability.clamp(
            min=self.epsilon, max=1.0 - self.epsilon
        )
        pixel_weights = self.class_weights[target]
        cross_entropy_term = (
            pixel_weights
            * torch.pow(-torch.log(true_probability), self.cross_entropy_gamma)
        ).mean()

        total = (
            self.dice_weight * dice_term
            + self.cross_entropy_weight * cross_entropy_term
        )
        return total, {
            "ell_dice": dice_term.detach(),
            "ell_cross_entropy": cross_entropy_term.detach(),
        }


def soft_transition_probability_matrix(
    probabilities: torch.Tensor, smoothing: float = 0.0
) -> torch.Tensor:
    """从softmax概率构造可微的批次垂向转移概率矩阵。"""

    if probabilities.ndim != 4:
        raise ValueError("probabilities应为[B,C,H,W]")
    if probabilities.shape[2] < 2:
        raise ValueError("深度方向至少需要2个像素")
    upper = probabilities[:, :, :-1, :]
    lower = probabilities[:, :, 1:, :]
    counts = torch.einsum("bihw,bjhw->ij", upper, lower)
    if smoothing:
        counts = counts + smoothing
    return counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)


def hard_transition_probability_matrix(
    target: torch.Tensor, num_classes: int, smoothing: float = 0.0
) -> torch.Tensor:
    """从整数标签构造批次真实VTPM；该矩阵不参与梯度计算。"""

    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
    return soft_transition_probability_matrix(one_hot.float(), smoothing=smoothing)


def vtpm_alignment_loss(
    logits: torch.Tensor, target: torch.Tensor, smoothing: float = 0.0
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """计算预测VTPM与标签VTPM的逐元素平方差之和。"""

    # 完整剖面的相邻像素数超过FP16最大有限值，因此整个VTPM统计块固定使用FP32。
    with torch.autocast(device_type=logits.device.type, enabled=False):
        probabilities = F.softmax(logits.float(), dim=1)
        prediction_vtpm = soft_transition_probability_matrix(
            probabilities, smoothing=smoothing
        )
        with torch.no_grad():
            target_vtpm = hard_transition_probability_matrix(
                target, logits.shape[1], smoothing=smoothing
            )
        difference = prediction_vtpm - target_vtpm
        loss = torch.sum(difference.square())
    return loss, {
        "prediction_vtpm": prediction_vtpm.detach(),
        "target_vtpm": target_vtpm.detach(),
        "vtpm_frobenius": torch.linalg.vector_norm(difference).detach(),
    }
