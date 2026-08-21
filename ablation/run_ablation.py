"""训练并评估ELL/VTPM的2×2组件消融实验。"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ablation.losses import ExponentialLogarithmicLoss, vtpm_alignment_loss
from ablation.model import UNet


VARIANTS = {
    "baseline": {"use_ell": False, "use_vtpm": False, "label": "CE"},
    "ell": {"use_ell": True, "use_vtpm": False, "label": "ELL"},
    "vtpm": {"use_ell": False, "use_vtpm": True, "label": "CE+VTPM"},
    "full": {"use_ell": True, "use_vtpm": True, "label": "ELL+VTPM"},
}

PALETTE = np.asarray(
    [
        [35, 92, 167],
        [125, 180, 213],
        [219, 241, 247],
        [255, 219, 124],
        [252, 120, 59],
        [208, 10, 0],
    ],
    dtype=np.uint8,
)


class F3SectionDataset(Dataset):
    """按二维剖面读取F3体数据，并统一为[深度, 水平]方向。"""

    def __init__(
        self,
        seismic_path: str,
        label_path: str,
        orientation: str,
        indices: Optional[Sequence[int]] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.orientation = orientation
        self.output_size = output_size
        self.seismic = np.load(seismic_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        if self.seismic.shape != self.labels.shape:
            raise ValueError(
                "地震数据和标签尺寸不一致：%s != %s"
                % (self.seismic.shape, self.labels.shape)
            )
        if orientation == "inline":
            section_count = self.seismic.shape[0]
        elif orientation == "crossline":
            section_count = self.seismic.shape[1]
        else:
            raise ValueError("orientation只能是inline或crossline")
        self.indices = list(range(section_count)) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        section_index = self.indices[item]
        if self.orientation == "inline":
            seismic_section = self.seismic[section_index].T
            label_section = self.labels[section_index].T
        else:
            seismic_section = self.seismic[:, section_index, :].T
            label_section = self.labels[:, section_index, :].T
        seismic_array = np.array(seismic_section, dtype=np.float32, copy=True)
        label_array = np.array(label_section, dtype=np.int64, copy=True)
        seismic_tensor = torch.from_numpy(seismic_array).unsqueeze(0)
        label_tensor = torch.from_numpy(label_array)
        if self.output_size is not None:
            seismic_tensor = torch.nn.functional.interpolate(
                seismic_tensor.unsqueeze(0),
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            label_tensor = torch.nn.functional.interpolate(
                label_tensor[None, None].float(),
                size=self.output_size,
                mode="nearest",
            ).squeeze(0).squeeze(0).long()
        return seismic_tensor, label_tensor, section_index


@dataclass
class MetricAccumulator:
    """累计分割混淆矩阵及垂向转移统计。"""

    num_classes: int

    def __post_init__(self) -> None:
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.prediction_transitions = np.zeros_like(self.confusion)
        self.target_transitions = np.zeros_like(self.confusion)
        self.reverse_count = 0
        self.transition_count = 0
        self.pixel_count = 0

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        if prediction.shape != target.shape:
            raise ValueError("预测和标签尺寸不一致")
        encoded = target.ravel() * self.num_classes + prediction.ravel()
        self.confusion += np.bincount(
            encoded, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.prediction_transitions += transition_counts(prediction, self.num_classes)
        self.target_transitions += transition_counts(target, self.num_classes)
        upper = prediction[:-1, :]
        lower = prediction[1:, :]
        self.reverse_count += int(np.count_nonzero(lower < upper))
        self.transition_count += int(upper.size)
        self.pixel_count += int(prediction.size)

    def merge(self, other: "MetricAccumulator") -> None:
        self.confusion += other.confusion
        self.prediction_transitions += other.prediction_transitions
        self.target_transitions += other.target_transitions
        self.reverse_count += other.reverse_count
        self.transition_count += other.transition_count
        self.pixel_count += other.pixel_count

    def metrics(self, smoothing: float = 0.0) -> Dict[str, float]:
        diagonal = np.diag(self.confusion).astype(np.float64)
        target_total = self.confusion.sum(axis=1).astype(np.float64)
        prediction_total = self.confusion.sum(axis=0).astype(np.float64)
        total = float(self.confusion.sum())
        class_accuracy = safe_divide(diagonal, target_total)
        union = target_total + prediction_total - diagonal
        iou = safe_divide(diagonal, union)
        frequencies = target_total / total
        prediction_vtpm = normalize_transition_counts(
            self.prediction_transitions, smoothing
        )
        target_vtpm = normalize_transition_counts(self.target_transitions, smoothing)
        difference = prediction_vtpm - target_vtpm
        return {
            "pixel_accuracy": float(diagonal.sum() / total),
            "mean_class_accuracy": float(np.nanmean(class_accuracy)),
            "mean_iou": float(np.nanmean(iou)),
            "frequency_weighted_iou": float(np.nansum(frequencies * iou)),
            "reverse_transition_rate": float(
                self.reverse_count / self.transition_count
            ),
            "vtpm_frobenius_to_target": float(np.linalg.norm(difference)),
            "vtpm_mse_to_target": float(np.mean(difference ** 2)),
            "num_pixels": self.pixel_count,
            "num_transitions": self.transition_count,
        }


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    return np.divide(numerator, denominator, out=result, where=denominator > 0)


def transition_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = labels[:-1, :].ravel() * num_classes + labels[1:, :].ravel()
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )


def normalize_transition_counts(counts: np.ndarray, smoothing: float) -> np.ndarray:
    values = counts.astype(np.float64) + smoothing
    row_sums = values.sum(axis=1, keepdims=True)
    return np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_split(num_sections: int, validation_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    generator = np.random.RandomState(seed)
    indices = generator.permutation(num_sections).tolist()
    validation_count = max(1, int(round(num_sections * validation_ratio)))
    return sorted(indices[validation_count:]), sorted(indices[:validation_count])


def compute_class_weights(
    label_path: str, indices: Iterable[int], num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.load(label_path, mmap_mode="r")
    counts = np.zeros(num_classes, dtype=np.int64)
    for index in indices:
        counts += np.bincount(labels[index].ravel(), minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("训练划分中存在没有样本的类别：%s" % counts.tolist())
    weights = np.sqrt(counts.sum() / counts.astype(np.float64))
    return counts, weights.astype(np.float32)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def primary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    use_ell: bool,
    ell_loss: ExponentialLogarithmicLoss,
    ce_loss: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if use_ell:
        return ell_loss(logits, target)
    loss = ce_loss(logits, target)
    return loss, {"cross_entropy": loss.detach()}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ell_loss: ExponentialLogarithmicLoss,
    ce_loss: nn.Module,
    use_ell: bool,
    use_vtpm: bool,
    lambda_vtpm: float,
    vtpm_smoothing: float,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()
    totals = {"total": 0.0, "primary": 0.0, "vtpm": 0.0}
    sample_count = 0
    for inputs, target, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss_main, _ = primary_loss(logits, target, use_ell, ell_loss, ce_loss)
            if use_vtpm:
                loss_vtpm, _ = vtpm_alignment_loss(
                    logits, target, smoothing=vtpm_smoothing
                )
            else:
                loss_vtpm = logits.new_zeros(())
            loss = loss_main + lambda_vtpm * loss_vtpm
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = inputs.shape[0]
        sample_count += batch_size
        totals["total"] += float(loss.detach()) * batch_size
        totals["primary"] += float(loss_main.detach()) * batch_size
        totals["vtpm"] += float(loss_vtpm.detach()) * batch_size
    return {name: value / sample_count for name, value in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    ell_loss: ExponentialLogarithmicLoss,
    ce_loss: nn.Module,
    use_ell: bool,
    use_vtpm: bool,
    lambda_vtpm: float,
    vtpm_smoothing: float,
    num_classes: int,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    accumulator = MetricAccumulator(num_classes)
    total_loss = 0.0
    sample_count = 0
    for inputs, target, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
            loss_main, _ = primary_loss(logits, target, use_ell, ell_loss, ce_loss)
            if use_vtpm:
                loss_vtpm, _ = vtpm_alignment_loss(
                    logits, target, smoothing=vtpm_smoothing
                )
            else:
                loss_vtpm = logits.new_zeros(())
            loss = loss_main + lambda_vtpm * loss_vtpm
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        targets = target.cpu().numpy()
        for prediction, label in zip(predictions, targets):
            accumulator.update(prediction, label)
        batch_size = inputs.shape[0]
        sample_count += batch_size
        total_loss += float(loss) * batch_size
    metrics = accumulator.metrics(vtpm_smoothing)
    return total_loss / sample_count, metrics


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    amp_enabled: bool,
    output_directory: str,
    dataset_name: str,
    saved_sections: Sequence[int],
) -> MetricAccumulator:
    model.eval()
    accumulator = MetricAccumulator(num_classes)
    os.makedirs(output_directory, exist_ok=True)
    saved_set = set(saved_sections)
    for inputs, target, section_indices in loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(inputs)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        targets = target.numpy()
        for prediction, label, section_index in zip(
            predictions, targets, section_indices.tolist()
        ):
            accumulator.update(prediction, label)
            if section_index in saved_set:
                image = Image.fromarray(PALETTE[prediction], mode="RGB")
                image.save(
                    os.path.join(
                        output_directory, "%s_%d.png" % (dataset_name, section_index)
                    )
                )
    return accumulator


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_variant(
    variant_name: str,
    args: argparse.Namespace,
    datasets: Dict[str, Dataset],
    class_weights: np.ndarray,
    device: torch.device,
) -> Tuple[nn.Module, List[Dict[str, object]], float]:
    settings = VARIANTS[variant_name]
    set_random_seed(args.seed)
    model = UNet(
        in_channels=1,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
    ).to(device)
    ell_loss = ExponentialLogarithmicLoss(
        torch.from_numpy(class_weights),
        dice_weight=args.ell_dice_weight,
        cross_entropy_weight=args.ell_ce_weight,
        dice_gamma=args.ell_dice_gamma,
        cross_entropy_gamma=args.ell_ce_gamma,
    ).to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=3
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = make_loader(
        datasets["train"], args.batch_size, True, args.num_workers, args.seed, device
    )
    validation_loader = make_loader(
        datasets["validation"],
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
        device,
    )

    variant_directory = os.path.join(args.output_dir, variant_name)
    os.makedirs(variant_directory, exist_ok=True)
    last_path = os.path.join(variant_directory, "last.pt")
    best_path = os.path.join(variant_directory, "best.pt")
    start_epoch = 0
    best_miou = -1.0
    history = []
    if args.resume and os.path.isfile(last_path):
        checkpoint = load_checkpoint(last_path, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint["best_miou"])
        history = checkpoint.get("history", [])

    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            ell_loss,
            ce_loss,
            bool(settings["use_ell"]),
            bool(settings["use_vtpm"]),
            args.lambda_vtpm,
            args.vtpm_smoothing,
            device,
            amp_enabled,
        )
        validation_loss, validation_metrics = validate(
            model,
            validation_loader,
            ell_loss,
            ce_loss,
            bool(settings["use_ell"]),
            bool(settings["use_vtpm"]),
            args.lambda_vtpm,
            args.vtpm_smoothing,
            args.num_classes,
            device,
            amp_enabled,
        )
        scheduler.step(train_metrics["total"])
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["total"],
            "train_primary_loss": train_metrics["primary"],
            "train_vtpm_loss": train_metrics["vtpm"],
            "validation_loss": validation_loss,
            "validation_miou": validation_metrics["mean_iou"],
            "validation_pa": validation_metrics["pixel_accuracy"],
        }
        history.append(record)
        if validation_metrics["mean_iou"] > best_miou:
            best_miou = validation_metrics["mean_iou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_miou": best_miou,
                    "variant": variant_name,
                    "settings": settings,
                },
                best_path,
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_miou": best_miou,
                "history": history,
            },
            last_path,
        )
        write_csv(os.path.join(variant_directory, "history.csv"), history)
        print(
            "[%s] Epoch %d/%d train=%.4f val=%.4f val_MIoU=%.4f"
            % (
                settings["label"],
                epoch + 1,
                args.epochs,
                train_metrics["total"],
                validation_loss,
                validation_metrics["mean_iou"],
            ),
            flush=True,
        )

    best_checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(best_checkpoint["model"])
    return model, history, time.time() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ELL/VTPM组件消融实验")
    parser.add_argument("--data-root", default="./data/f3_model")
    parser.add_argument("--output-dir", default="./ablation/outputs")
    parser.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--train-height", type=int, default=256)
    parser.add_argument("--train-width", type=int, default=192)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lambda-vtpm", type=float, default=0.1)
    parser.add_argument("--vtpm-smoothing", type=float, default=1e-6)
    parser.add_argument("--ell-dice-weight", type=float, default=0.8)
    parser.add_argument("--ell-ce-weight", type=float, default=0.2)
    parser.add_argument("--ell-dice-gamma", type=float, default=0.3)
    parser.add_argument("--ell-ce-gamma", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--saved-sections", default="0,99,199")
    parser.add_argument("--max-train-sections", type=int)
    parser.add_argument("--max-validation-sections", type=int)
    parser.add_argument("--max-test-sections", type=int)
    parser.set_defaults(amp=True)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用CUDA，但当前PyTorch无法访问GPU")
    return device


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    train_seismic = os.path.join(args.data_root, "train_seismic.npy")
    train_labels = os.path.join(args.data_root, "train_labels.npy")
    section_count = int(np.load(train_labels, mmap_mode="r").shape[0])
    train_indices, validation_indices = build_split(
        section_count, args.validation_ratio, args.seed
    )
    if args.max_train_sections:
        train_indices = train_indices[: args.max_train_sections]
    if args.max_validation_sections:
        validation_indices = validation_indices[: args.max_validation_sections]

    class_counts, class_weights = compute_class_weights(
        train_labels, train_indices, args.num_classes
    )
    test1_count = int(
        np.load(os.path.join(args.data_root, "test1_labels.npy"), mmap_mode="r").shape[0]
    )
    test2_count = int(
        np.load(os.path.join(args.data_root, "test2_labels.npy"), mmap_mode="r").shape[1]
    )
    test1_indices = list(range(test1_count))
    test2_indices = list(range(test2_count))
    if args.max_test_sections:
        test1_indices = test1_indices[: args.max_test_sections]
        test2_indices = test2_indices[: args.max_test_sections]

    datasets = {
        "train": F3SectionDataset(
            train_seismic,
            train_labels,
            "inline",
            train_indices,
            output_size=(args.train_height, args.train_width),
        ),
        "validation": F3SectionDataset(
            train_seismic,
            train_labels,
            "inline",
            validation_indices,
            output_size=(args.train_height, args.train_width),
        ),
        "test1": F3SectionDataset(
            os.path.join(args.data_root, "test1_seismic.npy"),
            os.path.join(args.data_root, "test1_labels.npy"),
            "inline",
            test1_indices,
        ),
        "test2": F3SectionDataset(
            os.path.join(args.data_root, "test2_seismic.npy"),
            os.path.join(args.data_root, "test2_labels.npy"),
            "crossline",
            test2_indices,
        ),
    }
    configuration = vars(args).copy()
    configuration.update(
        {
            "device_resolved": str(device),
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "class_counts": class_counts.tolist(),
            "class_weights": class_weights.tolist(),
        }
    )
    with open(
        os.path.join(args.output_dir, "configuration.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(configuration, handle, ensure_ascii=False, indent=2)

    saved_sections = [int(value) for value in args.saved_sections.split(",") if value]
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")
    rows = []
    summary = {"configuration": configuration, "variants": {}}
    # 分批运行部分变体时保留其他变体的既有汇总，当前变体将在本次评估后替换。
    if args.resume and os.path.isfile(metrics_path):
        with open(metrics_path, "r", encoding="utf-8-sig", newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row.get("variant") not in args.variants
            ]
    if args.resume and os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            previous_summary = json.load(handle)
        summary["variants"] = {
            name: value
            for name, value in previous_summary.get("variants", {}).items()
            if name not in args.variants
        }
    for variant_name in args.variants:
        model, history, training_seconds = train_variant(
            variant_name, args, datasets, class_weights, device
        )
        variant_directory = os.path.join(args.output_dir, variant_name)
        prediction_directory = os.path.join(variant_directory, "predictions")
        accumulators = {}
        for dataset_name in ("test1", "test2"):
            loader = make_loader(
                datasets[dataset_name],
                args.batch_size,
                False,
                args.num_workers,
                args.seed,
                device,
            )
            accumulators[dataset_name] = evaluate_dataset(
                model,
                loader,
                args.num_classes,
                device,
                args.amp and device.type == "cuda",
                prediction_directory,
                dataset_name,
                saved_sections,
            )
        overall = MetricAccumulator(args.num_classes)
        overall.merge(accumulators["test1"])
        overall.merge(accumulators["test2"])
        variant_summary = {
            "best_validation_miou": max(
                float(item["validation_miou"]) for item in history
            ),
            "training_seconds": training_seconds,
            "datasets": {},
        }
        for dataset_name, accumulator in (
            ("test1", accumulators["test1"]),
            ("test2", accumulators["test2"]),
            ("overall", overall),
        ):
            metrics = accumulator.metrics(args.vtpm_smoothing)
            row = {
                "variant": variant_name,
                "loss": VARIANTS[variant_name]["label"],
                "use_ell": VARIANTS[variant_name]["use_ell"],
                "use_vtpm": VARIANTS[variant_name]["use_vtpm"],
                "dataset": dataset_name,
            }
            row.update(metrics)
            rows.append(row)
            variant_summary["datasets"][dataset_name] = metrics
        summary["variants"][variant_name] = variant_summary
        write_csv(metrics_path, rows)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("消融实验完成，结果位于：%s" % os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
