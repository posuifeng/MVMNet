from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from pointmatch.fusion_data import MultiViewRegistrationSegmentationDataset
from pointmatch.fusion_model import MVMNet, compute_multiview_fusion_losses
from pointmatch.segmentation_model import compute_segmentation_metrics

HEAD_SPECS = [
    ("empty_aux1_to_target", "empty", 1, 0),
    ("empty_aux2_to_target", "empty", 2, 0),
    ("full_aux1_to_target", "full", 1, 0),
    ("full_aux2_to_target", "full", 2, 0),
]
VIEW_NAMES = ["target_lidar2", "aux1_lidar1", "aux2_lidar3"]
PARAM_NAMES = ["a", "b", "c", "tx", "ty", "tz"]


def parse_int_tuple(
    value: object, default: tuple[int, ...] = (8, 16, 32)
) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return tuple((int(v) for v in value))
    return tuple((int(item.strip()) for item in str(value).split(",") if item.strip()))


def make_loader(
    dataset: MultiViewRegistrationSegmentationDataset, batch_size: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def to_device(
    batch: dict[str, torch.Tensor | str | list[str]], device: torch.device
) -> dict[str, torch.Tensor | str | list[str]]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def masked_seg_metrics(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    valid = mask > 0.5
    if not bool(torch.any(valid)):
        return {
            "accuracy": float("nan"),
            "mean_iou": float("nan"),
            "box_iou": float("nan"),
            "surface_iou": float("nan"),
        }
    return {
        key: float(value)
        for key, value in compute_segmentation_metrics(
            logits[valid], labels[valid]
        ).items()
    }


def head_param_mae(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    valid = mask > 0.5
    if not bool(torch.any(valid)):
        return {}
    diff = torch.abs(pred[valid] - target[valid]).detach().cpu().numpy()
    names = ["a", "b", "c", "tx", "ty", "tz"]
    out = {"reg_param_mae": float(diff.mean())}
    for idx, name in enumerate(names):
        out[f"reg_param_mae_{name}"] = float(diff[:, idx].mean())
    return out


def class_confusion(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    valid = mask > 0.5
    if valid.ndim == 1:
        valid = valid[:, None].expand_as(labels)
    elif valid.ndim == 2 and valid.shape[1] == 1:
        valid = valid.expand_as(labels)
    elif valid.ndim != labels.ndim:
        valid = valid.reshape(labels.shape)
    pred = torch.argmax(logits, dim=-1)
    pred_valid = pred[valid].detach().cpu().numpy().astype(np.int64)
    label_valid = labels[valid].detach().cpu().numpy().astype(np.int64)
    if len(label_valid) == 0:
        return {
            "tp_box": 0.0,
            "fp_box": 0.0,
            "fn_box": 0.0,
            "tn_surface": 0.0,
            "pred_box_ratio": float("nan"),
            "true_box_ratio": float("nan"),
        }
    tp = float(np.sum((pred_valid == 1) & (label_valid == 1)))
    fp = float(np.sum((pred_valid == 1) & (label_valid == 0)))
    fn = float(np.sum((pred_valid == 0) & (label_valid == 1)))
    tn = float(np.sum((pred_valid == 0) & (label_valid == 0)))
    return {
        "tp_box": tp,
        "fp_box": fp,
        "fn_box": fn,
        "tn_surface": tn,
        "pred_box_ratio": float(np.mean(pred_valid == 1)),
        "true_box_ratio": float(np.mean(label_valid == 1)),
    }


def binary_metrics_from_counts(
    tp: float, fp: float, fn: float, tn: float
) -> dict[str, float]:
    box_iou = tp / max(tp + fp + fn, 1.0)
    surface_iou = tn / max(tn + fp + fn, 1.0)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "accuracy": float(accuracy),
        "mean_iou": float(0.5 * (box_iou + surface_iou)),
        "box_iou": float(box_iou),
        "surface_iou": float(surface_iou),
        "box_precision": float(precision),
        "box_recall": float(recall),
    }


def symmetric_nn_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    trim_ratio: float,
    distance_scale: torch.Tensor | None = None,
) -> dict[str, float]:
    dists = torch.cdist(pred, target)
    pred_to_target = dists.min(dim=2).values
    target_to_pred = dists.min(dim=1).values
    keep_p = max(1, int(pred_to_target.shape[1] * trim_ratio))
    keep_t = max(1, int(target_to_pred.shape[1] * trim_ratio))
    pred_trim = torch.topk(pred_to_target, k=keep_p, dim=1, largest=False).values
    target_trim = torch.topk(target_to_pred, k=keep_t, dim=1, largest=False).values
    out = {
        "nn_pred_to_target": float(pred_to_target.mean().detach().cpu()),
        "nn_target_to_pred": float(target_to_pred.mean().detach().cpu()),
        "symmetric_nn": float(
            (0.5 * (pred_to_target.mean(dim=1) + target_to_pred.mean(dim=1)))
            .mean()
            .detach()
            .cpu()
        ),
        "trimmed_symmetric_nn": float(
            (0.5 * (pred_trim.mean(dim=1) + target_trim.mean(dim=1)))
            .mean()
            .detach()
            .cpu()
        ),
        "p95_pred_to_target": float(
            torch.quantile(pred_to_target.reshape(-1), 0.95).detach().cpu()
        ),
        "p95_target_to_pred": float(
            torch.quantile(target_to_pred.reshape(-1), 0.95).detach().cpu()
        ),
    }
    if distance_scale is not None:
        scale = distance_scale.reshape(-1, 1).to(device=pred.device, dtype=pred.dtype)
        pred_to_target_mm = pred_to_target * scale
        target_to_pred_mm = target_to_pred * scale
        pred_trim_mm = pred_trim * scale
        target_trim_mm = target_trim * scale
        out.update(
            {
                "nn_pred_to_target_mm": float(pred_to_target_mm.mean().detach().cpu()),
                "nn_target_to_pred_mm": float(target_to_pred_mm.mean().detach().cpu()),
                "symmetric_nn_mm": float(
                    (
                        0.5
                        * (
                            pred_to_target_mm.mean(dim=1)
                            + target_to_pred_mm.mean(dim=1)
                        )
                    )
                    .mean()
                    .detach()
                    .cpu()
                ),
                "trimmed_symmetric_nn_mm": float(
                    (0.5 * (pred_trim_mm.mean(dim=1) + target_trim_mm.mean(dim=1)))
                    .mean()
                    .detach()
                    .cpu()
                ),
                "p95_pred_to_target_mm": float(
                    torch.quantile(pred_to_target_mm.reshape(-1), 0.95).detach().cpu()
                ),
                "p95_target_to_pred_mm": float(
                    torch.quantile(target_to_pred_mm.reshape(-1), 0.95).detach().cpu()
                ),
            }
        )
    return out


def build_model_from_checkpoint(checkpoint, device):
    cfg = checkpoint.get("args", {})
    expected = {
        "fusion_model_name": "cross_attention_multiview",
        "encoder_name": "dgcnn",
        "cross_num_tokens": 256,
        "registration_head_type": "shared_pairwise_conditioned_moe",
        "use_normals": False,
        "use_volume_token": True,
        "use_registered_seg_aggregation": True,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(
                f"Checkpoint architecture mismatch: {key}={cfg.get(key)!r}; expected {value!r}"
            )
    model = MVMNet().to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def evaluate_split(
    model: torch.nn.Module,
    manifest_path: Path,
    split_path: Path,
    device: torch.device,
    num_points: int,
    batch_size: int,
    num_workers: int,
    param_weights: torch.Tensor,
    trim_ratio: float,
    reg_geom_weight: float,
    reg_param_weight: float,
    reg_transform_weight: float,
    seg_weight: float,
    seg_consistency_weight: float,
    seg_consistency_chunk_size: int,
    max_batches: int,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    dataset = MultiViewRegistrationSegmentationDataset(
        manifest_path=manifest_path,
        split_path=split_path,
        num_points=num_points,
        augment=False,
    )
    loader = make_loader(dataset, batch_size=batch_size, num_workers=num_workers)
    totals: dict[str, float] = {
        "loss": 0.0,
        "reg_param_loss": 0.0,
        "reg_transform_loss": 0.0,
        "reg_geom_loss": 0.0,
        "seg_loss": 0.0,
        "seg_consistency_loss": 0.0,
        "seg_consistency_valid": 0.0,
        "seg_accuracy": 0.0,
        "seg_mean_iou": 0.0,
        "seg_box_iou": 0.0,
        "seg_surface_iou": 0.0,
        "empty_seg_mean_iou": 0.0,
        "full_seg_mean_iou": 0.0,
    }
    param_totals: dict[str, float] = {}
    seg_confusion: dict[str, dict[str, float]] = {}
    head_totals: dict[str, dict[str, float]] = {}
    rows: list[dict[str, float | str]] = []
    count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            tensor_batch = to_device(batch, device)
            outputs = model(
                tensor_batch["empty_points"],
                tensor_batch["full_points"],
                volume_m3=tensor_batch.get("volume_m3"),
                volume_quality=tensor_batch.get("volume_quality"),
                volume_geometry=tensor_batch.get("volume_geometry"),
                volume_mask=tensor_batch.get("volume_mask"),
            )
            _, metrics = compute_multiview_fusion_losses(
                batch=tensor_batch,
                outputs=outputs,
                param_weights=param_weights,
                trim_ratio=trim_ratio,
                reg_geom_weight=reg_geom_weight,
                reg_param_weight=reg_param_weight,
                reg_transform_weight=reg_transform_weight,
                seg_weight=seg_weight,
                seg_consistency_weight=seg_consistency_weight,
                seg_consistency_chunk_size=seg_consistency_chunk_size,
            )
            batch_size_actual = int(tensor_batch["empty_points"].shape[0])
            for key in totals:
                if key in metrics:
                    totals[key] += float(metrics[key]) * batch_size_actual
            empty_metrics = masked_seg_metrics(
                outputs["empty_seg_logits"].detach(),
                tensor_batch["empty_seg_labels"],
                tensor_batch["empty_seg_mask"],
            )
            full_metrics = masked_seg_metrics(
                outputs["full_seg_logits"].detach(),
                tensor_batch["full_seg_labels"],
                tensor_batch["full_seg_mask"],
            )
            totals["empty_seg_mean_iou"] += (
                float(empty_metrics["mean_iou"]) * batch_size_actual
            )
            totals["full_seg_mean_iou"] += (
                float(full_metrics["mean_iou"]) * batch_size_actual
            )
            param_metrics = head_param_mae(
                outputs["reg_params"],
                tensor_batch["reg_head_params"],
                tensor_batch["reg_head_mask"],
            )
            for key, value in param_metrics.items():
                param_totals[key] = (
                    param_totals.get(key, 0.0) + float(value) * batch_size_actual
                )
            group_ids = batch["group_id"]
            if isinstance(group_ids, str):
                group_ids = [group_ids]
            pred_classes_empty = torch.argmax(outputs["empty_seg_logits"], dim=-1)
            pred_classes_full = torch.argmax(outputs["full_seg_logits"], dim=-1)
            for state_name, logits_key, labels_key, mask_key in [
                ("empty", "empty_seg_logits", "empty_seg_labels", "empty_seg_mask"),
                ("full", "full_seg_logits", "full_seg_labels", "full_seg_mask"),
            ]:
                logits = outputs[logits_key].detach()
                labels = tensor_batch[labels_key]
                view_mask = tensor_batch[mask_key]
                for view_idx, view_name in enumerate(VIEW_NAMES):
                    key = f"{state_name}_{view_name}"
                    counts = class_confusion(
                        logits[:, view_idx, :, :],
                        labels[:, view_idx, :],
                        view_mask[:, view_idx : view_idx + 1],
                    )
                    bucket = seg_confusion.setdefault(
                        key,
                        {
                            "tp_box": 0.0,
                            "fp_box": 0.0,
                            "fn_box": 0.0,
                            "tn_surface": 0.0,
                        },
                    )
                    for count_key in ["tp_box", "fp_box", "fn_box", "tn_surface"]:
                        bucket[count_key] += counts[count_key]
            for head_idx, (head_name, state_name, source_idx, target_idx) in enumerate(
                HEAD_SPECS
            ):
                pred_params = outputs["reg_params"][:, head_idx, :]
                target_params = tensor_batch["reg_head_params"][:, head_idx, :]
                mask = tensor_batch["reg_head_mask"][:, head_idx] > 0.5
                if not bool(torch.any(mask)):
                    continue
                if state_name == "empty":
                    pred_points = outputs["registered_empty_points"][
                        :, source_idx, :, :
                    ]
                    target_points = tensor_batch["empty_points"][:, target_idx, :, :]
                    distance_scale = tensor_batch["empty_target_scale"].reshape(-1)
                else:
                    pred_points = outputs["registered_full_points"][:, source_idx, :, :]
                    target_points = tensor_batch["full_points"][:, target_idx, :, :]
                    distance_scale = tensor_batch["full_target_scale"].reshape(-1)
                pred_points = pred_points[mask]
                target_points = target_points[mask]
                distance_scale = distance_scale[mask]
                param_abs = torch.abs(pred_params[mask] - target_params[mask])
                geom = symmetric_nn_metrics(
                    pred_points,
                    target_points,
                    trim_ratio=trim_ratio,
                    distance_scale=distance_scale,
                )
                l2_indexed = float(
                    torch.linalg.norm(pred_points - target_points, dim=-1)
                    .mean()
                    .detach()
                    .cpu()
                )
                l2_indexed_mm = float(
                    (
                        torch.linalg.norm(pred_points - target_points, dim=-1)
                        * distance_scale.reshape(-1, 1)
                    )
                    .mean()
                    .detach()
                    .cpu()
                )
                bucket = head_totals.setdefault(head_name, {"count": 0.0})
                batch_valid = float(int(mask.sum().item()))
                bucket["count"] += batch_valid
                bucket["indexed_l2"] = (
                    bucket.get("indexed_l2", 0.0) + l2_indexed * batch_valid
                )
                bucket["indexed_l2_mm"] = (
                    bucket.get("indexed_l2_mm", 0.0) + l2_indexed_mm * batch_valid
                )
                for metric_key, value in geom.items():
                    bucket[metric_key] = (
                        bucket.get(metric_key, 0.0) + float(value) * batch_valid
                    )
                bucket["reg_param_mae"] = (
                    bucket.get("reg_param_mae", 0.0)
                    + float(param_abs.mean().detach().cpu()) * batch_valid
                )
                for param_idx, param_name in enumerate(PARAM_NAMES):
                    bucket[f"reg_param_mae_{param_name}"] = (
                        bucket.get(f"reg_param_mae_{param_name}", 0.0)
                        + float(param_abs[:, param_idx].mean().detach().cpu())
                        * batch_valid
                    )
            for sample_idx, group_id in enumerate(group_ids):
                sample_param = head_param_mae(
                    outputs["reg_params"][sample_idx : sample_idx + 1],
                    tensor_batch["reg_head_params"][sample_idx : sample_idx + 1],
                    tensor_batch["reg_head_mask"][sample_idx : sample_idx + 1],
                )
                sample_head_geom: dict[str, float] = {}
                for head_idx, (
                    head_name,
                    state_name,
                    source_idx,
                    target_idx,
                ) in enumerate(HEAD_SPECS):
                    if (
                        float(
                            tensor_batch["reg_head_mask"][sample_idx, head_idx]
                            .detach()
                            .cpu()
                        )
                        <= 0.5
                    ):
                        continue
                    if state_name == "empty":
                        pred_points_i = outputs["registered_empty_points"][
                            sample_idx : sample_idx + 1, source_idx, :, :
                        ]
                        target_points_i = tensor_batch["empty_points"][
                            sample_idx : sample_idx + 1, target_idx, :, :
                        ]
                        distance_scale_i = tensor_batch["empty_target_scale"][
                            sample_idx : sample_idx + 1
                        ].reshape(-1)
                    else:
                        pred_points_i = outputs["registered_full_points"][
                            sample_idx : sample_idx + 1, source_idx, :, :
                        ]
                        target_points_i = tensor_batch["full_points"][
                            sample_idx : sample_idx + 1, target_idx, :, :
                        ]
                        distance_scale_i = tensor_batch["full_target_scale"][
                            sample_idx : sample_idx + 1
                        ].reshape(-1)
                    geom_i = symmetric_nn_metrics(
                        pred_points_i,
                        target_points_i,
                        trim_ratio=trim_ratio,
                        distance_scale=distance_scale_i,
                    )
                    sample_head_geom[f"{head_name}_trimmed_symmetric_nn"] = geom_i[
                        "trimmed_symmetric_nn"
                    ]
                    sample_head_geom[f"{head_name}_trimmed_symmetric_nn_mm"] = geom_i[
                        "trimmed_symmetric_nn_mm"
                    ]
                rows.append(
                    {
                        "group_id": str(group_id),
                        "empty_pred_box_ratio": float(
                            (pred_classes_empty[sample_idx] == 1).float().mean().item()
                        ),
                        "empty_true_box_ratio": float(
                            (tensor_batch["empty_seg_labels"][sample_idx] == 1)
                            .float()
                            .mean()
                            .item()
                        ),
                        "full_pred_box_ratio": float(
                            (pred_classes_full[sample_idx] == 1).float().mean().item()
                        ),
                        "full_true_box_ratio": float(
                            (tensor_batch["full_seg_labels"][sample_idx] == 1)
                            .float()
                            .mean()
                            .item()
                        ),
                        **sample_param,
                        **sample_head_geom,
                    }
                )
            count += batch_size_actual
            if max_batches > 0 and batch_index >= max_batches:
                break
    summary = {key: value / max(count, 1) for key, value in totals.items()}
    for key, value in param_totals.items():
        summary[key] = value / max(count, 1)
    for key, counts in seg_confusion.items():
        metrics = binary_metrics_from_counts(
            tp=counts["tp_box"],
            fp=counts["fp_box"],
            fn=counts["fn_box"],
            tn=counts["tn_surface"],
        )
        for metric_key, value in metrics.items():
            summary[f"{key}_{metric_key}"] = value
    for head_name, metrics in head_totals.items():
        head_count = max(metrics.get("count", 0.0), 1.0)
        for metric_key, value in metrics.items():
            if metric_key == "count":
                summary[f"{head_name}_count"] = value
            else:
                summary[f"{head_name}_{metric_key}"] = value / head_count
    summary["sample_count"] = float(count)
    return (summary, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multiview fusion registration + segmentation checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "artifacts"
            / "checkpoints"
            / "fusion_reg_seg_multiview_cluster_v1"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--labels-dir",
        default="",
        help="Defaults to the labels_dir stored in the checkpoint; falls back to multiview_fusion_cluster_top1_v1 only if absent.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Evaluate train, val, and test splits.",
    )
    parser.add_argument(
        "--output-dir", default="", help="Defaults to checkpoint clean_review."
    )
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = dict(checkpoint.get("args", {}))
    labels_dir = Path(
        args.labels_dir
        or cfg.get(
            "labels_dir", ROOT / "artifacts" / "multiview_fusion_cluster_top1_v1"
        )
    )
    manifest_path = labels_dir / "manifest.jsonl"
    model = build_model_from_checkpoint(checkpoint, device)
    param_weights = torch.tensor(
        [
            float(cfg.get("weight_a", 1.0)),
            float(cfg.get("weight_b", 3.0)),
            float(cfg.get("weight_c", 1.0)),
            float(cfg.get("weight_tx", 2.5)),
            float(cfg.get("weight_ty", 1.0)),
            float(cfg.get("weight_tz", 3.0)),
        ],
        dtype=torch.float32,
        device=device,
    )
    split_names = ["train", "val", "test"] if args.all_splits else [args.split]
    all_summaries: dict[str, dict[str, float | str | int]] = {}
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent / "clean_review"
    )
    for split_name in split_names:
        split_path = labels_dir / "splits" / f"{split_name}.txt"
        output_dir = output_root / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        summary, rows = evaluate_split(
            model=model,
            manifest_path=manifest_path,
            split_path=split_path,
            device=device,
            num_points=int(args.num_points),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            param_weights=param_weights,
            trim_ratio=float(cfg.get("trim_ratio", 0.9)),
            reg_geom_weight=float(cfg.get("reg_geom_weight", 5.0)),
            reg_param_weight=float(cfg.get("reg_param_weight", 1.0)),
            reg_transform_weight=float(cfg.get("reg_transform_weight", 2.0)),
            seg_weight=float(cfg.get("seg_weight", 1.0)),
            seg_consistency_weight=float(cfg.get("seg_consistency_weight", 0.0)),
            seg_consistency_chunk_size=int(cfg.get("seg_consistency_chunk_size", 1024)),
            max_batches=int(args.max_batches),
        )
        summary.update(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
                "labels_dir": str(labels_dir),
                "split": split_name,
                "num_points": int(args.num_points),
                "batch_size": int(args.batch_size),
                "device": str(device),
                "registration_distance_unit": "mm",
                "registration_distance_coordinate_space": "rescaled_z_point_coordinates",
                "registration_distance_note": "Trimmed symmetric NN *_mm metrics are normalized distances multiplied by each sample target_scale after project z is rescaled.",
                "encoder_name": str(cfg.get("encoder_name", "pointnet")),
                "encoder_knn_k": int(cfg.get("encoder_knn_k", 16)),
                "encoder_multi_scale_k": list(
                    parse_int_tuple(cfg.get("encoder_multi_scale_k", (8, 16, 32)))
                ),
            }
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if rows:
            with (output_dir / "per_sample_metrics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                fieldnames = sorted({key for row in rows for key in row})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        all_summaries[split_name] = summary
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    if len(all_summaries) > 1:
        root_output = output_root
        root_output.mkdir(parents=True, exist_ok=True)
        (root_output / "all_splits_summary.json").write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
