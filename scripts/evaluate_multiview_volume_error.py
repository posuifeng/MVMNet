from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_multiview_fusion_registration_segmentation import (  # noqa: E402
    build_model_from_checkpoint,
    make_loader,
    to_device,
)
from pointmatch.fusion_data import (
    MultiViewRegistrationSegmentationDataset,
)  # noqa: E402
from pointmatch.fusion_model import (  # noqa: E402
    _apply_volume_label_geometry,
    _denormalize_state_points,
    _soft_surface_volume_m3,
)


def predict_volume_m3(
    batch: dict[str, torch.Tensor | str | list[str]],
    outputs: dict[str, torch.Tensor],
    sigma_mm: float,
    max_points: int,
) -> torch.Tensor:
    geometry = batch["volume_geometry"].to(
        dtype=outputs["registered_empty_points"].dtype,
        device=outputs["registered_empty_points"].device,
    )
    empty_world = _denormalize_state_points(
        outputs["registered_empty_points"],
        batch["empty_target_center"],
        batch["empty_target_scale"],
    )
    full_world = _denormalize_state_points(
        outputs["registered_full_points"],
        batch["full_target_center"],
        batch["full_target_scale"],
    )
    empty_volume_space = _apply_volume_label_geometry(
        empty_world.flatten(1, 2), geometry, is_full=False
    )
    full_volume_space = _apply_volume_label_geometry(
        full_world.flatten(1, 2), geometry, is_full=True
    )

    empty_surface_prob = torch.softmax(outputs["empty_seg_logits"], dim=-1)[..., 0]
    full_surface_prob = torch.softmax(outputs["full_seg_logits"], dim=-1)[..., 0]
    empty_valid = (
        batch["empty_valid_mask"].unsqueeze(-1).to(dtype=empty_surface_prob.dtype)
    )
    full_valid = (
        batch["full_valid_mask"].unsqueeze(-1).to(dtype=full_surface_prob.dtype)
    )
    empty_weights = (empty_surface_prob * empty_valid).flatten(1)
    full_weights = (full_surface_prob * full_valid).flatten(1)

    grid_step = geometry[:, 7].clamp_min(1.0)
    usable_grid_samples = geometry[:, 9].clamp_min(1.0)
    domain_area_mm2 = usable_grid_samples * grid_step.square()
    return _soft_surface_volume_m3(
        empty_points=empty_volume_space,
        full_points=full_volume_space,
        empty_weights=empty_weights,
        full_weights=full_weights,
        domain_area_mm2=domain_area_mm2,
        sigma_mm=float(sigma_mm),
        max_points=int(max_points),
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    output_dir: Path,
    labels_dir_override: Path | None = None,
) -> dict[str, float | str | int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = dict(checkpoint.get("args", {}))
    labels_dir = labels_dir_override or Path(
        cfg.get(
            "labels_dir", ROOT / "artifacts" / "multiview_fusion_cluster_top1_volume_v1"
        )
    )
    manifest_path = labels_dir / "manifest.jsonl"
    split_path = labels_dir / "splits" / f"{split}.txt"
    model = build_model_from_checkpoint(checkpoint, device)

    dataset = MultiViewRegistrationSegmentationDataset(
        manifest_path=manifest_path,
        split_path=split_path,
        num_points=int(cfg.get("ori_num_points", 4096)),
        augment=False,
    )
    loader = make_loader(dataset, batch_size=batch_size, num_workers=num_workers)

    rows: list[dict[str, float | str]] = []
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            tensor_batch = to_device(batch, device)
            outputs = model(
                tensor_batch["empty_points"],
                tensor_batch["full_points"],
                volume_m3=tensor_batch.get("volume_m3"),
                volume_quality=tensor_batch.get("volume_quality"),
                volume_geometry=tensor_batch.get("volume_geometry"),
                volume_mask=tensor_batch.get("volume_mask"),
            )
            pred = predict_volume_m3(
                tensor_batch,
                outputs,
                sigma_mm=float(cfg.get("volume_loss_sigma_mm", 100.0)),
                max_points=int(cfg.get("volume_loss_max_points", 1024)),
            )
            target = (
                tensor_batch["volume_m3"]
                .view(-1)
                .to(dtype=pred.dtype, device=pred.device)
            )
            mask = tensor_batch["volume_mask"].view(-1) > 0.5
            for idx in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
                pred_i = float(pred[idx].detach().cpu())
                target_i = float(target[idx].detach().cpu())
                abs_i = abs(pred_i - target_i)
                rel_i = abs_i / max(abs(target_i), 1e-6)
                group_id = (
                    tensor_batch["group_id"][idx]
                    if isinstance(tensor_batch["group_id"], list)
                    else str(idx)
                )
                rows.append(
                    {
                        "group_id": str(group_id),
                        "pred_volume_m3": pred_i,
                        "gt_volume_m3": target_i,
                        "abs_error_m3": abs_i,
                        "rel_error": rel_i,
                    }
                )
                preds.append(pred_i)
                targets.append(target_i)

    pred_t = torch.tensor(preds, dtype=torch.float64)
    target_t = torch.tensor(targets, dtype=torch.float64)
    diff = pred_t - target_t
    abs_diff = diff.abs()
    rel = abs_diff / target_t.abs().clamp_min(1e-6)
    summary: dict[str, float | str | int] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": split,
        "sample_count": int(len(rows)),
        "pred_volume_m3_mean": float(pred_t.mean()),
        "gt_volume_m3_mean": float(target_t.mean()),
        "volume_bias_m3_mean": float(diff.mean()),
        "volume_mae_m3": float(abs_diff.mean()),
        "volume_rmse_m3": float(torch.sqrt((diff.square()).mean())),
        "volume_rel_mae": float(rel.mean()),
        "volume_rel_rmse": float(torch.sqrt((rel.square()).mean())),
        "volume_abs_error_m3_median": float(abs_diff.median()),
        "volume_abs_error_m3_p95": float(torch.quantile(abs_diff, 0.95)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "volume_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "volume_per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "pred_volume_m3",
                "gt_volume_m3",
                "abs_error_m3",
                "rel_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate predicted-vs-GT volume error for a multiview checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--labels-dir",
        default="",
        help="Override checkpoint labels_dir for manifest/splits.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = evaluate_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        split=args.split,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=torch.device(args.device),
        output_dir=Path(args.output_dir),
        labels_dir_override=Path(args.labels_dir) if args.labels_dir else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
