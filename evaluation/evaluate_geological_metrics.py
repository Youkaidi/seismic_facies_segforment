"""
批量评估`.npy`预测剖面的命令行入口
"""

import argparse
import csv
import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from evaluation.geological_metrics import (
    evaluate_geological_consistency,
)
from evaluation.segmentation_metrics import runningScore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "评估分割精度、反向层序转移率以及预测与真实VTPM的距离。"
        )
    )
    parser.add_argument(
        "--prediction-dir", default="./runs_11/preds_test_npy"
    )
    parser.add_argument(
        "--label-volume", default="./data/f3_model/test1_labels.npy"
    )
    parser.add_argument(
        "--output-dir", default="./result/geological_metrics/runs_11_core"
    )
    parser.add_argument(
        "--method-name",
        default=None,
        help="写入aggregate_metrics.csv的方法名称，默认使用预测目录名。",
    )
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--vertical-axis", type=int, default=-1)
    parser.add_argument("--vtpm-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--filename-regex",
        default=r"^slice_(\d+)_pred\.npy$",
        help="第一个正则捕获组必须是标签数据体中的剖面索引。",
    )
    return parser.parse_args()


def segmentation_scores(target: np.ndarray, prediction: np.ndarray, num_classes: int) -> Dict[str, float]:
    """计算常用图像分割指标。"""

    score = runningScore(num_classes)
    score.update([target], [prediction])
    values, _ = score.get_scores()
    return {
        "pixel_accuracy": float(values["Pixel Acc: "]),
        "mean_class_accuracy": float(values["Mean Class Acc: "]),
        "mean_iou": float(values["Mean IoU: "]),
        "frequency_weighted_iou": float(values["Freq Weighted IoU: "]),
    }


def find_predictions(prediction_dir: str, filename_regex: str):
    """查找预测文件，并从文件名中解析剖面索引。"""

    pattern = re.compile(filename_regex)
    found = []
    for name in sorted(os.listdir(prediction_dir)):
        match = pattern.match(name)
        if match:
            found.append((int(match.group(1)), name))
    if not found:
        raise RuntimeError(
            "目录%s中没有与%s匹配的预测文件"
            % (prediction_dir, filename_regex)
        )
    return found


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    """将指标记录写入带UTF-8 BOM的CSV，便于Excel直接打开。"""

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_vtpms(path: str, matrices: Dict[str, np.ndarray]) -> None:
    """绘制测试标签和预测结果的VTPM对比图。"""

    names = ["target_vtpm", "prediction_vtpm"]
    titles = ["Test ground truth", "Prediction"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    image = None
    for axis, name, title in zip(axes, names, titles):
        image = axis.imshow(matrices[name], vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_title(title)
        axis.set_xlabel("Lower facies")
        axis.set_ylabel("Upper facies")
        axis.set_xticks(range(matrices[name].shape[1]))
        axis.set_yticks(range(matrices[name].shape[0]))
        for row in range(matrices[name].shape[0]):
            for col in range(matrices[name].shape[1]):
                axis.text(
                    col,
                    row,
                    "%.3f" % matrices[name][row, col],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrices[name][row, col] > 0.5 else "black",
                )
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_sections(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """计算剖面级指标的均值与总体标准差。"""

    statistics = {}
    excluded = {"section_index", "prediction_file", "num_transitions", "num_traces"}
    for key in rows[0]:
        if key in excluded:
            continue
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        statistics[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return statistics


def main() -> None:
    """执行完整评估并保存CSV、JSON、矩阵和可视化结果。"""

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    method_name = args.method_name or os.path.basename(
        os.path.normpath(args.prediction_dir)
    )

    label_volume = np.load(args.label_volume, mmap_mode="r")
    predictions = []
    targets = []
    per_section = []
    for section_index, filename in find_predictions(
        args.prediction_dir, args.filename_regex
    ):
        if section_index < 0 or section_index >= label_volume.shape[0]:
            raise IndexError(
                "文件%s解析出的剖面索引%d超出标签数据体范围（共%d个剖面）"
                % (filename, section_index, label_volume.shape[0])
            )
        prediction = np.load(os.path.join(args.prediction_dir, filename)).astype(
            np.int64, copy=False
        )
        target = np.asarray(label_volume[section_index]).astype(np.int64, copy=False)
        geological, _ = evaluate_geological_consistency(
            prediction,
            target,
            num_classes=args.num_classes,
            vertical_axis=args.vertical_axis,
            vtpm_smoothing=args.vtpm_smoothing,
        )
        row = {
            "section_index": section_index,
            "prediction_file": filename,
        }
        row.update(segmentation_scores(target, prediction, args.num_classes))
        row.update(geological)
        per_section.append(row)
        predictions.append(prediction)
        targets.append(target)

    prediction_stack = np.stack(predictions)
    target_stack = np.stack(targets)
    aggregate_geological, matrices = evaluate_geological_consistency(
        prediction_stack,
        target_stack,
        num_classes=args.num_classes,
        vertical_axis=args.vertical_axis,
        vtpm_smoothing=args.vtpm_smoothing,
    )
    aggregate = segmentation_scores(
        target_stack, prediction_stack, args.num_classes
    )
    aggregate.update(aggregate_geological)

    write_csv(os.path.join(args.output_dir, "per_section_metrics.csv"), per_section)
    aggregate_row = {
        "method": method_name,
        "num_sections": len(per_section),
    }
    aggregate_row.update(aggregate)
    write_csv(
        os.path.join(args.output_dir, "aggregate_metrics.csv"), [aggregate_row]
    )
    with open(
        os.path.join(args.output_dir, "summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "method": method_name,
                "configuration": vars(args),
                "num_sections": len(per_section),
                "aggregate": aggregate,
                "per_section_statistics": summarize_sections(per_section),
                "per_section": per_section,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    np.savez_compressed(
        os.path.join(args.output_dir, "transition_matrices.npz"), **matrices
    )
    plot_vtpms(os.path.join(args.output_dir, "vtpm_comparison.png"), matrices)

    print("Evaluated %d sections" % len(per_section))
    for key in (
        "pixel_accuracy",
        "mean_class_accuracy",
        "frequency_weighted_iou",
        "reverse_transition_rate",
        "vtpm_frobenius_to_target",
    ):
        print("%-40s %.6f" % (key, aggregate[key]))
    print("Results saved to %s" % os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
