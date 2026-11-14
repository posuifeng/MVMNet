from __future__ import annotations
import torch


def compute_segmentation_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    metrics = {"accuracy": correct / max(total, 1)}
    eps = 1e-06
    for class_id, class_name in [(0, "surface"), (1, "box")]:
        pred_mask = preds == class_id
        true_mask = labels == class_id
        intersection = torch.logical_and(pred_mask, true_mask).sum().item()
        union = torch.logical_or(pred_mask, true_mask).sum().item()
        pred_count = pred_mask.sum().item()
        true_count = true_mask.sum().item()
        precision = intersection / max(pred_count, 1)
        recall = intersection / max(true_count, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, eps)
        iou = intersection / max(union, 1)
        metrics[f"{class_name}_iou"] = iou
        metrics[f"{class_name}_f1"] = f1
    metrics["mean_iou"] = 0.5 * (metrics["surface_iou"] + metrics["box_iou"])
    return metrics
