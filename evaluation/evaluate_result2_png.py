"""批量评估Result2中保存为RGB PNG的地震相分割结果。"""

import argparse
import csv
import json
import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from evaluation.geological_metrics import (
    adjacent_transition_pairs,
    evaluate_geological_consistency,
    transition_count_matrix,
    transition_probability_matrix,
)
from evaluation.segmentation_metrics import runningScore


# Result2中存在两套色板。下面的映射已通过同名标签PNG与原始F3标签逐像素核对。
COLOR_TO_CLASS = {
    (35, 92, 167): 0,
    (125, 180, 213): 1,
    (219, 241, 247): 2,
    (255, 219, 124): 3,
    (252, 120, 59): 4,
    (208, 10, 0): 5,
    (0, 66, 153): 0,
    (104, 168, 206): 1,
    (213, 239, 246): 2,
    (255, 213, 103): 3,
    (252, 99, 27): 4,
}

MODEL_DIRECTORIES = {
    "Baseline": "对照",
    "DGCNN": "DGCNN",
    "SegFormer": "Segformer",
    "U-Net": "U-Net",
    "MC-Net": "MC_Net",
}

SECTION_PATTERN = re.compile(r"^(test[12])_(\d+)\.png$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量评估Result2中Baseline、DGCNN、SegFormer、U-Net和MC-Net的PNG预测结果。"
    )
    parser.add_argument("--result-root", default="./Result2")
    parser.add_argument("--output-dir", default="./Result2/evaluation")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--vtpm-smoothing", type=float, default=0.0)
    return parser.parse_args()


def decode_segmentation_png(path: str) -> np.ndarray:
    """将RGB分割图严格转换为整数类别矩阵。"""

    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    labels = np.full(rgb.shape[:2], -1, dtype=np.int64)
    for color, class_index in COLOR_TO_CLASS.items():
        labels[np.all(rgb == color, axis=-1)] = class_index

    if np.any(labels < 0):
        unknown = np.unique(rgb[labels < 0], axis=0)
        raise ValueError(
            "文件%s包含未定义颜色：%s" % (path, unknown.tolist())
        )
    return labels


def discover_label_sections(label_dir: str) -> Dict[Tuple[str, int], str]:
    """读取标签目录中的test1/test2剖面文件。"""

    sections = {}
    for filename in sorted(os.listdir(label_dir)):
        match = SECTION_PATTERN.match(filename)
        if not match:
            continue
        key = (match.group(1).lower(), int(match.group(2)))
        sections[key] = os.path.join(label_dir, filename)
    if not sections:
        raise RuntimeError("标签目录%s中没有找到可评估的PNG剖面" % label_dir)
    return sections


def discover_model_sections(
    model_dir: str, expected_keys: Iterable[Tuple[str, int]]
) -> Dict[Tuple[str, int], str]:
    """查找模型预测，并验证其与标签剖面一一对应。"""

    sections = {}
    for filename in sorted(os.listdir(model_dir)):
        match = SECTION_PATTERN.match(filename)
        if not match:
            continue
        key = (match.group(1).lower(), int(match.group(2)))
        sections[key] = os.path.join(model_dir, filename)

    expected = set(expected_keys)
    actual = set(sections)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            "目录%s中的预测文件与标签不一致；缺少=%s，额外=%s"
            % (model_dir, missing, extra)
        )
    return sections


def segmentation_metrics(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]], num_classes: int
) -> Dict[str, float]:
    """在若干预测—标签对上累计混淆矩阵并计算分割指标。"""

    scorer = runningScore(num_classes)
    for prediction, target in pairs:
        scorer.update([target], [prediction])
    values, _ = scorer.get_scores()
    return {
        "pixel_accuracy": float(values["Pixel Acc: "]),
        "mean_class_accuracy": float(values["Mean Class Acc: "]),
        "mean_iou": float(values["Mean IoU: "]),
        "frequency_weighted_iou": float(values["Freq Weighted IoU: "]),
    }


def aggregate_geological_metrics(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    vtpm_smoothing: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """对不同尺寸的多个剖面累计转移计数并计算地质指标。"""

    prediction_counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    target_counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    reverse_count = 0
    transition_count = 0
    trace_count = 0

    for prediction, target in pairs:
        # PNG数组的第0维是深度方向，第1维是水平方向。
        upper, lower, _ = adjacent_transition_pairs(
            prediction, num_classes, vertical_axis=0
        )
        reverse_count += int(np.count_nonzero(lower < upper))
        transition_count += int(upper.size)
        trace_count += int(upper.shape[0])
        prediction_counts += transition_count_matrix(
            prediction, num_classes, vertical_axis=0
        )
        target_counts += transition_count_matrix(
            target, num_classes, vertical_axis=0
        )

    prediction_vtpm = transition_probability_matrix(
        prediction_counts, smoothing=vtpm_smoothing
    )
    target_vtpm = transition_probability_matrix(
        target_counts, smoothing=vtpm_smoothing
    )
    difference = prediction_vtpm - target_vtpm
    metrics = {
        "reverse_transition_rate": reverse_count / transition_count,
        "vtpm_frobenius_to_target": float(np.linalg.norm(difference)),
        "vtpm_mse_to_target": float(np.mean(difference ** 2)),
        "num_pixels": int(sum(prediction.size for prediction, _ in pairs)),
        "num_transitions": transition_count,
        "num_traces": trace_count,
    }
    matrices = {
        "prediction_counts": prediction_counts,
        "target_counts": target_counts,
        "prediction_vtpm": prediction_vtpm,
        "target_vtpm": target_vtpm,
    }
    return metrics, matrices


def evaluate_pairs(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
    num_classes: int,
    vtpm_smoothing: float,
) -> Dict[str, float]:
    """合并计算分割指标与地质合理性指标。"""

    metrics = segmentation_metrics(pairs, num_classes)
    geological, _ = aggregate_geological_metrics(
        pairs, num_classes, vtpm_smoothing
    )
    metrics.update(geological)
    return metrics


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    """将字典列表写入CSV。"""

    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    label_dir = os.path.join(args.result_root, "labels", "label")
    label_paths = discover_label_sections(label_dir)
    decoded_labels = {
        key: decode_segmentation_png(path) for key, path in label_paths.items()
    }

    aggregate_rows = []
    per_section_rows = []
    summary = {"configuration": vars(args), "models": {}}

    for method, directory_name in MODEL_DIRECTORIES.items():
        model_dir = os.path.join(args.result_root, directory_name)
        prediction_paths = discover_model_sections(model_dir, label_paths.keys())
        decoded_predictions = {
            key: decode_segmentation_png(path)
            for key, path in prediction_paths.items()
        }

        pairs_by_dataset = {"test1": [], "test2": []}
        for dataset, section_index in sorted(label_paths):
            prediction = decoded_predictions[(dataset, section_index)]
            target = decoded_labels[(dataset, section_index)]
            if prediction.shape != target.shape:
                raise ValueError(
                    "%s的%s_%d预测尺寸%s与标签尺寸%s不一致"
                    % (
                        method,
                        dataset,
                        section_index,
                        prediction.shape,
                        target.shape,
                    )
                )
            pairs_by_dataset[dataset].append((prediction, target))

            section_metrics = evaluate_pairs(
                [(prediction, target)], args.num_classes, args.vtpm_smoothing
            )
            section_row = {
                "method": method,
                "dataset": dataset,
                "section_index": section_index,
            }
            section_row.update(section_metrics)
            per_section_rows.append(section_row)

        method_summary = {}
        for dataset in ("test1", "test2", "overall"):
            pairs = (
                pairs_by_dataset["test1"] + pairs_by_dataset["test2"]
                if dataset == "overall"
                else pairs_by_dataset[dataset]
            )
            metrics = evaluate_pairs(
                pairs, args.num_classes, args.vtpm_smoothing
            )
            row = {
                "method": method,
                "dataset": dataset,
                "num_sections": len(pairs),
            }
            row.update(metrics)
            aggregate_rows.append(row)
            method_summary[dataset] = metrics
        summary["models"][method] = method_summary

    write_csv(os.path.join(args.output_dir, "aggregate_metrics.csv"), aggregate_rows)
    write_csv(os.path.join(args.output_dir, "per_section_metrics.csv"), per_section_rows)
    write_csv(
        os.path.join(args.output_dir, "overall_comparison.csv"),
        [row for row in aggregate_rows if row["dataset"] == "overall"],
    )
    with open(
        os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("已评估%d个模型、每个模型%d张剖面。" % (len(MODEL_DIRECTORIES), len(label_paths)))
    print("结果已保存至：%s" % os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
