"""用于有序地震相的地质一致性评价指标。"""
"""垂向转移统计和地质合理性指标"""

from typing import Dict, Tuple

import numpy as np


def _validate_labels(labels: np.ndarray, num_classes: int, name: str) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim < 1:
        raise ValueError("%s至少需要一个维度" % name)
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.round(labels)):
            raise ValueError("%s必须包含整数类别标签" % name)
        labels = labels.astype(np.int64)
    if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
        raise ValueError(
            "%s包含超出[0, %d]范围的标签" % (name, num_classes - 1)
        )
    return labels.astype(np.int64, copy=False)


def adjacent_transition_pairs(
    labels: np.ndarray, num_classes: int, vertical_axis: int = -1
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """返回上下相邻标签对及非垂向维度的形状。

    非垂向维度中的每个位置都视为一条地震道，返回的上下标签数组形状为
    ``[地震道数量, 深度采样点数量 - 1]``。
    """

    labels = _validate_labels(labels, num_classes, "标签")
    axis = vertical_axis if vertical_axis >= 0 else labels.ndim + vertical_axis
    if axis < 0 or axis >= labels.ndim:
        raise ValueError(
            "垂向轴%d不适用于%d维数组"
            % (vertical_axis, labels.ndim)
        )
    moved = np.moveaxis(labels, axis, -1)
    trace_shape = moved.shape[:-1]
    depth = moved.shape[-1]
    if depth < 2:
        raise ValueError("垂向轴至少需要包含两个采样点")
    traces = moved.reshape(-1, depth)
    return traces[:, :-1], traces[:, 1:], trace_shape


def transition_count_matrix(
    labels: np.ndarray, num_classes: int, vertical_axis: int = -1
) -> np.ndarray:
    """统计垂向上有方向的相邻类别转移次数。"""

    upper, lower, _ = adjacent_transition_pairs(labels, num_classes, vertical_axis)
    flat_index = num_classes * upper.ravel() + lower.ravel()
    return np.bincount(
        flat_index, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def transition_probability_matrix(
    counts: np.ndarray, smoothing: float = 0.0
) -> np.ndarray:
    """对转移计数矩阵按行归一化，并可选使用拉普拉斯平滑。"""

    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("转移计数必须是方阵")
    if np.any(counts < 0):
        raise ValueError("转移计数不能为负数")
    if smoothing < 0:
        raise ValueError("平滑系数不能为负数")

    smoothed = counts + float(smoothing)
    row_sums = smoothed.sum(axis=1, keepdims=True)
    return np.divide(
        smoothed,
        row_sums,
        out=np.zeros_like(smoothed, dtype=np.float64),
        where=row_sums > 0,
    )


def _rate(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def evaluate_geological_consistency(
    prediction: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    vertical_axis: int = -1,
    vtpm_smoothing: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """评价硬标签预测结果的地质一致性。

    指标说明
    --------
    reverse_transition_rate
        下部类别编号小于上部类别编号的转移比例。该指标假设类别编号按照
        从浅到深的地层顺序排列。
    vtpm_*_to_target
        行归一化后的预测VTPM与真实VTPM之间的距离。
    """

    prediction = _validate_labels(prediction, num_classes, "预测结果")
    target = _validate_labels(target, num_classes, "真实标签")
    if prediction.shape != target.shape:
        raise ValueError(
            "预测结果与真实标签形状不同：%s与%s"
            % (prediction.shape, target.shape)
        )

    pred_upper, pred_lower, _ = adjacent_transition_pairs(
        prediction, num_classes, vertical_axis
    )

    pred_reverse = pred_lower < pred_upper

    pred_counts = transition_count_matrix(prediction, num_classes, vertical_axis)
    target_counts = transition_count_matrix(target, num_classes, vertical_axis)
    pred_vtpm = transition_probability_matrix(pred_counts, vtpm_smoothing)
    target_vtpm = transition_probability_matrix(target_counts, vtpm_smoothing)

    target_diff = pred_vtpm - target_vtpm
    metrics = {
        "reverse_transition_rate": _rate(pred_reverse),
        "vtpm_frobenius_to_target": float(np.linalg.norm(target_diff)),
        "vtpm_mse_to_target": float(np.mean(target_diff ** 2)),
        "num_transitions": int(pred_reverse.size),
        "num_traces": int(pred_reverse.shape[0]),
    }
    matrices = {
        "prediction_counts": pred_counts,
        "target_counts": target_counts,
        "prediction_vtpm": pred_vtpm,
        "target_vtpm": target_vtpm,
    }
    return metrics, matrices
