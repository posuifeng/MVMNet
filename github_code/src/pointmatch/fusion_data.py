from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from .common import apply_transform, special_params_to_matrix_translation
from .training_data import normalize_pair, rescale_points_z, rescale_special_params_z


def read_split_ids(split_path: str | Path) -> list[str]:
    path = Path(split_path)
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    split_ids = [line.strip() for line in text.splitlines() if line.strip()]
    excluded_paths = [
        path.parent / "excluded_group_ids.txt",
        path.parent.parent / "excluded_group_ids.txt",
    ]
    excluded: set[str] = set()
    for excluded_path in excluded_paths:
        if excluded_path.exists():
            excluded.update(
                (
                    line.strip()
                    for line in excluded_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and (not line.lstrip().startswith("#"))
                )
            )
    if excluded:
        split_ids = [group_id for group_id in split_ids if group_id not in excluded]
    return split_ids


MULTIVIEW_STATES = ("empty", "full")
MULTIVIEW_VIEWS = ("target", "aux1", "aux2")
MULTIVIEW_VIEW_INDEX = {name: idx for idx, name in enumerate(MULTIVIEW_VIEWS)}
MULTIVIEW_STATE_INDEX = {name: idx for idx, name in enumerate(MULTIVIEW_STATES)}


@dataclass
class MultiViewSegPaths:
    box_path: str
    surface_path: str


@dataclass
class MultiViewItem:
    view_id: str
    points_path: str
    points_format: str
    seg: MultiViewSegPaths | None


@dataclass
class MultiViewRegistrationTarget:
    head_id: str
    time_state: str
    source_key: str
    target_key: str
    transform_type: str
    param_order: list[str]
    param_values: list[float] | None
    mean_error: float | None
    label_source: str


@dataclass
class MultiViewSample:
    group_id: str
    scene_id: str
    empty_views: dict[str, MultiViewItem]
    full_views: dict[str, MultiViewItem]
    registration_targets: list[MultiViewRegistrationTarget]
    meta: dict[str, object]


def _parse_multiview_item(row: dict) -> MultiViewItem:
    seg = None
    box_path = str(row.get("seg_box_path", "")).strip()
    surface_path = str(row.get("seg_surface_path", "")).strip()
    if box_path and surface_path:
        seg = MultiViewSegPaths(box_path=box_path, surface_path=surface_path)
    return MultiViewItem(
        view_id=str(row.get("view_id", "")),
        points_path=str(row["points_path"]),
        points_format=str(row.get("points_format", "xyzi")),
        seg=seg,
    )


def _parse_multiview_view_group(row: dict) -> dict[str, MultiViewItem]:
    return {
        name: _parse_multiview_item(row[name])
        for name in MULTIVIEW_VIEWS
        if name in row
    }


def _parse_registration_target(row: dict) -> MultiViewRegistrationTarget:
    params = row.get("param_values")
    return MultiViewRegistrationTarget(
        head_id=str(row["head_id"]),
        time_state=str(row["time_state"]),
        source_key=str(row["source_key"]),
        target_key=str(row["target_key"]),
        transform_type=str(row.get("transform_type", "special_6")),
        param_order=[
            str(v) for v in row.get("param_order", ["a", "b", "c", "tx", "ty", "tz"])
        ],
        param_values=None if params is None else [float(v) for v in params],
        mean_error=None if row.get("mean_error") is None else float(row["mean_error"]),
        label_source=str(row.get("label_source", "")),
    )


def load_multiview_manifest(manifest_path: str | Path) -> dict[str, MultiViewSample]:
    samples: dict[str, MultiViewSample] = {}
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            group_id = str(row["group_id"])
            samples[group_id] = MultiViewSample(
                group_id=group_id,
                scene_id=str(row.get("scene_id", group_id)),
                empty_views=_parse_multiview_view_group(row.get("empty", {})),
                full_views=_parse_multiview_view_group(row.get("full", {})),
                registration_targets=[
                    _parse_registration_target(item)
                    for item in row.get("registration_targets", [])
                ],
                meta=dict(row.get("meta", {})),
            )
    return samples


def _parse_volume_meta(meta: dict[str, object]) -> dict[str, np.ndarray]:
    volume = meta.get("volume") if isinstance(meta, dict) else None
    if not isinstance(volume, dict):
        return {
            "volume_mm3": np.array([0.0], dtype=np.float32),
            "volume_m3": np.array([0.0], dtype=np.float32),
            "volume_mask": np.array([0.0], dtype=np.float32),
            "volume_quality": np.zeros(3, dtype=np.float32),
            "volume_geometry": np.zeros(10, dtype=np.float32),
        }
    empty_z = volume.get("empty_z_transform")
    full_z = volume.get("full_z_transform")
    full_align = volume.get("full_alignment_to_empty")
    empty_z = empty_z if isinstance(empty_z, dict) else {}
    full_z = full_z if isinstance(full_z, dict) else {}
    full_align = full_align if isinstance(full_align, dict) else {}
    return {
        "volume_mm3": np.array(
            [float(volume.get("total_volume_mm3", 0.0))], dtype=np.float32
        ),
        "volume_m3": np.array(
            [float(volume.get("total_volume_m3", 0.0))], dtype=np.float32
        ),
        "volume_mask": np.array([1.0], dtype=np.float32),
        "volume_quality": np.array(
            [
                float(volume.get("missing_ratio_of_empty_domain", 1.0)),
                float(volume.get("extrapolated_ratio_of_usable", 1.0)),
                float(volume.get("mean_x_distance_mm", 0.0)),
            ],
            dtype=np.float32,
        ),
        "volume_geometry": np.array(
            [
                float(empty_z.get("z_scale", 1.0)),
                float(empty_z.get("z_offset", 0.0)),
                float(full_z.get("z_scale", 1.0)),
                float(full_z.get("z_offset", 0.0)),
                float(full_align.get("x_shift", 0.0)),
                float(full_align.get("y_shift", 0.0)),
                float(full_align.get("z_shift", 0.0)),
                float(volume.get("grid_step_mm", 10.0)),
                float(volume.get("empty_domain_grid_samples", 0.0)),
                float(
                    volume.get(
                        "usable_grid_samples",
                        volume.get("empty_domain_grid_samples", 0.0),
                    )
                ),
            ],
            dtype=np.float32,
        ),
    }


def _sample_or_pad_rows(
    points: np.ndarray, num_points: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= 0:
        raise ValueError("Cannot sample from an empty point cloud.")
    if len(points) == num_points:
        idx = np.arange(len(points), dtype=np.int64)
        return (points, idx)
    if len(points) > num_points:
        idx = rng.choice(len(points), size=num_points, replace=False).astype(np.int64)
        return (points[idx], idx)
    extra = rng.choice(len(points), size=num_points - len(points), replace=True).astype(
        np.int64
    )
    idx = np.concatenate([np.arange(len(points), dtype=np.int64), extra], axis=0)
    return (points[idx], idx)


def _balanced_sample_indices(
    labels: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    box_sample_ratio: float,
    min_box_points: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) <= 0:
        raise ValueError("Cannot sample from empty labels.")
    box_idx = np.flatnonzero(labels == 1)
    surface_idx = np.flatnonzero(labels == 0)
    if len(box_idx) == 0 or len(surface_idx) == 0:
        return _sample_or_pad_rows(
            np.zeros((len(labels), 1), dtype=np.float32), num_points, rng
        )[1]
    target_box = int(round(float(num_points) * float(box_sample_ratio)))
    target_box = max(int(min_box_points), target_box)
    target_box = min(max(target_box, 1), num_points - 1)
    target_surface = num_points - target_box
    sampled_box = rng.choice(
        box_idx, size=target_box, replace=len(box_idx) < target_box
    )
    sampled_surface = rng.choice(
        surface_idx, size=target_surface, replace=len(surface_idx) < target_surface
    )
    sampled = np.concatenate(
        [sampled_box.astype(np.int64), sampled_surface.astype(np.int64)], axis=0
    )
    rng.shuffle(sampled)
    return sampled.astype(np.int64)


def _natural_sample_indices_with_rare_supplement(
    labels: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
    rare_label: int = 1,
    min_rare_points: int = 64,
    max_rare_ratio: float = 0.02,
) -> np.ndarray:
    sampled_idx = _sample_or_pad_rows(
        np.zeros((len(labels), 1), dtype=np.float32), num_points, rng
    )[1]
    labels = np.asarray(labels, dtype=np.int64)
    rare_idx = np.flatnonzero(labels == int(rare_label))
    common_idx = np.flatnonzero(labels != int(rare_label))
    if len(rare_idx) == 0 or len(common_idx) == 0:
        return sampled_idx
    target_rare = min(
        int(min_rare_points), int(np.ceil(float(num_points) * float(max_rare_ratio)))
    )
    target_rare = min(max(target_rare, 0), num_points)
    if target_rare <= 0:
        return sampled_idx
    sampled_rare_mask = labels[sampled_idx] == int(rare_label)
    current_rare = int(np.sum(sampled_rare_mask))
    if current_rare >= target_rare:
        return sampled_idx
    replace_positions = np.flatnonzero(~sampled_rare_mask)
    if len(replace_positions) == 0:
        return sampled_idx
    add_count = min(target_rare - current_rare, len(replace_positions))
    chosen_positions = rng.choice(replace_positions, size=add_count, replace=False)
    sampled_idx[chosen_positions] = rng.choice(
        rare_idx, size=add_count, replace=len(rare_idx) < add_count
    )
    rng.shuffle(sampled_idx)
    return sampled_idx.astype(np.int64)


def _load_points(path: str | Path) -> np.ndarray:
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns in {path}, got shape {points.shape}"
        )
    return points


def _load_optional_seg_points(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.stat().st_size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points = np.loadtxt(path, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if points.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 columns in {path}, got shape {points.shape}"
        )
    return points[:, :3].astype(np.float32)


def _load_seg_supervision_raw(seg: MultiViewSegPaths) -> tuple[np.ndarray, np.ndarray]:
    box = _load_optional_seg_points(seg.box_path).astype(np.float32)
    surface = _load_optional_seg_points(seg.surface_path).astype(np.float32)
    if len(box) == 0 and len(surface) == 0:
        raise ValueError(
            f"Segmentation supervision is empty: {seg.surface_path}, {seg.box_path}"
        )
    all_points = np.vstack([surface, box]).astype(np.float32)
    all_labels = np.concatenate(
        [np.zeros(len(surface), dtype=np.int64), np.ones(len(box), dtype=np.int64)],
        axis=0,
    )
    return (all_points, all_labels)


def _label_raw_points_by_direct_gt(
    points_xyz_raw: np.ndarray, seg: MultiViewSegPaths | None
) -> tuple[np.ndarray, np.ndarray]:
    if seg is None:
        return (
            np.zeros(len(points_xyz_raw), dtype=np.int64),
            np.array([0.0], dtype=np.float32),
        )
    gt_points, gt_labels = _load_seg_supervision_raw(seg)
    tree = cKDTree(gt_points)
    _, idx = tree.query(points_xyz_raw.astype(np.float32), k=1, workers=-1)
    return (gt_labels[idx].astype(np.int64), np.array([1.0], dtype=np.float32))


def _apply_special_params_numpy(
    points_xyz: np.ndarray, params: list[float] | np.ndarray
) -> np.ndarray:
    matrix, translation = special_params_to_matrix_translation(
        np.asarray(params, dtype=np.float64)
    )
    return apply_transform(points_xyz.astype(np.float64), matrix, translation).astype(
        np.float32
    )


def _registration_params_by_source(
    registration_targets: list[MultiViewRegistrationTarget], state_name: str
) -> dict[str, list[float]]:
    params_by_source: dict[str, list[float]] = {}
    for target in registration_targets:
        if target.time_state != state_name or target.param_values is None:
            continue
        params_by_source[target.source_key] = target.param_values
    return params_by_source


class MultiViewRegistrationSegmentationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split_path: str | Path | None,
        num_points: int = 8192,
        augment: bool = False,
        seed: int = 42,
        balanced_seg_sampling: bool = False,
        box_sample_ratio: float = 0.125,
        min_box_points: int = 256,
        rare_box_supplement: bool = False,
        rare_box_min_points: int = 64,
        rare_box_max_ratio: float = 0.02,
    ) -> None:
        self.samples_by_id = load_multiview_manifest(manifest_path)
        split_ids = (
            list(self.samples_by_id.keys())
            if not split_path
            else read_split_ids(split_path)
        )
        self.samples = [
            self.samples_by_id[group_id]
            for group_id in split_ids
            if group_id in self.samples_by_id
        ]
        if not self.samples:
            raise ValueError(
                f"split has no usable samples: {split_path or manifest_path}"
            )
        self.num_points = int(num_points)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.balanced_seg_sampling = bool(balanced_seg_sampling)
        self.box_sample_ratio = float(box_sample_ratio)
        self.min_box_points = int(min_box_points)
        self.rare_box_supplement = bool(rare_box_supplement)
        self.rare_box_min_points = int(rare_box_min_points)
        self.rare_box_max_ratio = float(rare_box_max_ratio)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_view(
        self,
        item: MultiViewItem,
        target_center: np.ndarray,
        target_scale: float,
        rng: np.random.Generator,
        label_query_xyz_raw: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = _load_points(item.points_path)
        raw_xyz = rescale_points_z(raw[:, :3].astype(np.float32))
        if label_query_xyz_raw is None:
            label_query_xyz_raw = raw[:, :3].astype(np.float32)
        seg_labels_all, seg_mask = _label_raw_points_by_direct_gt(
            label_query_xyz_raw, item.seg
        )
        if self.balanced_seg_sampling and float(seg_mask[0]) > 0.5:
            sampled_idx = _balanced_sample_indices(
                labels=seg_labels_all,
                num_points=self.num_points,
                rng=rng,
                box_sample_ratio=self.box_sample_ratio,
                min_box_points=self.min_box_points,
            )
            sampled_xyz = raw_xyz[sampled_idx]
        elif self.rare_box_supplement and float(seg_mask[0]) > 0.5:
            sampled_idx = _natural_sample_indices_with_rare_supplement(
                labels=seg_labels_all,
                num_points=self.num_points,
                rng=rng,
                rare_label=1,
                min_rare_points=self.rare_box_min_points,
                max_rare_ratio=self.rare_box_max_ratio,
            )
            sampled_xyz = raw_xyz[sampled_idx]
        else:
            sampled_xyz, sampled_idx = _sample_or_pad_rows(
                raw_xyz, self.num_points, rng
            )
        sampled_seg = seg_labels_all[sampled_idx]
        norm_xyz = ((sampled_xyz - target_center[None, :]) / target_scale).astype(
            np.float32
        )
        view_center = raw_xyz.mean(axis=0).astype(np.float32)
        return (sampled_xyz, norm_xyz, sampled_seg, seg_mask, view_center)

    def _load_state_views(
        self,
        views: dict[str, MultiViewItem],
        rng: np.random.Generator,
        registration_targets: list[MultiViewRegistrationTarget],
        state_name: str,
    ) -> dict[str, object]:
        target_item = views["target"]
        target_raw = _load_points(target_item.points_path)
        target_xyz_all = rescale_points_z(target_raw[:, :3].astype(np.float32))
        target_center = target_xyz_all.mean(axis=0).astype(np.float32)
        target_scale = float(
            np.linalg.norm(target_xyz_all.max(axis=0) - target_xyz_all.min(axis=0))
            + 1e-06
        )
        params_by_source = _registration_params_by_source(
            registration_targets, state_name
        )
        raw_xyz_views: list[np.ndarray] = []
        norm_xyz_views: list[np.ndarray] = []
        seg_label_views: list[np.ndarray] = []
        seg_mask_views: list[np.ndarray] = []
        valid_mask_views: list[np.ndarray] = []
        view_centers: list[np.ndarray] = []
        all_xyz_by_view: dict[str, np.ndarray] = {}
        for view_name in MULTIVIEW_VIEWS:
            if view_name not in views:
                raw_xyz_views.append(np.zeros((self.num_points, 3), dtype=np.float32))
                norm_xyz_views.append(np.zeros((self.num_points, 3), dtype=np.float32))
                seg_label_views.append(np.zeros(self.num_points, dtype=np.int64))
                seg_mask_views.append(np.array([0.0], dtype=np.float32))
                valid_mask_views.append(np.array([0.0], dtype=np.float32))
                view_centers.append(np.zeros(3, dtype=np.float32))
                continue
            view_raw_xyz = _load_points(views[view_name].points_path)[:, :3].astype(
                np.float32
            )
            if view_name == "target":
                label_query_xyz_raw = view_raw_xyz
            else:
                params = params_by_source.get(view_name)
                label_query_xyz_raw = (
                    _apply_special_params_numpy(view_raw_xyz, params)
                    if params is not None
                    else view_raw_xyz
                )
            (
                sampled_raw,
                sampled_norm,
                sampled_seg,
                seg_mask,
                view_center,
            ) = self._load_view(
                item=views[view_name],
                target_center=target_center,
                target_scale=target_scale,
                rng=rng,
                label_query_xyz_raw=label_query_xyz_raw,
            )
            all_xyz_by_view[view_name] = rescale_points_z(view_raw_xyz)
            raw_xyz_views.append(sampled_raw)
            norm_xyz_views.append(sampled_norm)
            seg_label_views.append(sampled_seg)
            seg_mask_views.append(seg_mask)
            valid_mask_views.append(np.array([1.0], dtype=np.float32))
            view_centers.append(view_center)
        return {
            "points_xyz": np.stack(norm_xyz_views, axis=0).astype(np.float32),
            "points_xyz_raw": np.stack(raw_xyz_views, axis=0).astype(np.float32),
            "seg_labels": np.stack(seg_label_views, axis=0).astype(np.int64),
            "seg_mask": np.concatenate(seg_mask_views, axis=0).astype(np.float32),
            "valid_mask": np.concatenate(valid_mask_views, axis=0).astype(np.float32),
            "view_centers": np.stack(view_centers, axis=0).astype(np.float32),
            "target_center": target_center.astype(np.float32),
            "target_scale": np.array([target_scale], dtype=np.float32),
            "all_xyz_by_view": all_xyz_by_view,
        }

    def _build_registration_targets(
        self, sample: MultiViewSample, state_views: dict[str, dict[str, object]]
    ) -> dict[str, np.ndarray | list[str]]:
        head_ids: list[str] = []
        time_states: list[int] = []
        source_indices: list[int] = []
        target_indices: list[int] = []
        params: list[np.ndarray] = []
        raw_params: list[np.ndarray] = []
        masks: list[float] = []
        mean_errors: list[float] = []
        for item in sample.registration_targets:
            head_ids.append(item.head_id)
            time_states.append(MULTIVIEW_STATE_INDEX[item.time_state])
            source_indices.append(MULTIVIEW_VIEW_INDEX[item.source_key])
            target_indices.append(MULTIVIEW_VIEW_INDEX[item.target_key])
            if item.param_values is None:
                params.append(np.zeros(6, dtype=np.float32))
                raw_params.append(np.zeros(6, dtype=np.float32))
                masks.append(0.0)
            else:
                params_raw = rescale_special_params_z(
                    np.asarray(item.param_values, dtype=np.float32)
                )
                state_info = state_views[item.time_state]
                source_xyz = state_info["all_xyz_by_view"][item.source_key]
                target_xyz = state_info["all_xyz_by_view"][item.target_key]
                _, _, params_norm, _, _ = normalize_pair(
                    source_xyz, target_xyz, params_raw
                )
                params.append(params_norm.astype(np.float32))
                raw_params.append(params_raw.astype(np.float32))
                masks.append(1.0)
            mean_errors.append(float(item.mean_error or 0.0))
        if not head_ids:
            return {
                "head_ids": [],
                "time_state": np.zeros(0, dtype=np.int64),
                "source_indices": np.zeros(0, dtype=np.int64),
                "target_indices": np.zeros(0, dtype=np.int64),
                "params": np.zeros((0, 6), dtype=np.float32),
                "raw_params": np.zeros((0, 6), dtype=np.float32),
                "mask": np.zeros(0, dtype=np.float32),
                "mean_error": np.zeros(0, dtype=np.float32),
            }
        return {
            "head_ids": head_ids,
            "time_state": np.asarray(time_states, dtype=np.int64),
            "source_indices": np.asarray(source_indices, dtype=np.int64),
            "target_indices": np.asarray(target_indices, dtype=np.int64),
            "params": np.stack(params, axis=0).astype(np.float32),
            "raw_params": np.stack(raw_params, axis=0).astype(np.float32),
            "mask": np.asarray(masks, dtype=np.float32),
            "mean_error": np.asarray(mean_errors, dtype=np.float32),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | list[str]]:
        sample = self.samples[index]
        rng_seed = None if self.augment else self.seed + index
        rng = np.random.default_rng(rng_seed)
        empty_state = self._load_state_views(
            sample.empty_views, rng, sample.registration_targets, "empty"
        )
        full_state = self._load_state_views(
            sample.full_views, rng, sample.registration_targets, "full"
        )
        reg = self._build_registration_targets(
            sample, state_views={"empty": empty_state, "full": full_state}
        )
        volume = _parse_volume_meta(sample.meta)
        return {
            "group_id": sample.group_id,
            "scene_id": sample.scene_id,
            "empty_points": torch.from_numpy(empty_state["points_xyz"]),
            "empty_points_raw": torch.from_numpy(empty_state["points_xyz_raw"]),
            "empty_seg_labels": torch.from_numpy(empty_state["seg_labels"]),
            "empty_seg_mask": torch.from_numpy(empty_state["seg_mask"]),
            "empty_valid_mask": torch.from_numpy(empty_state["valid_mask"]),
            "full_points": torch.from_numpy(full_state["points_xyz"]),
            "full_points_raw": torch.from_numpy(full_state["points_xyz_raw"]),
            "full_seg_labels": torch.from_numpy(full_state["seg_labels"]),
            "full_seg_mask": torch.from_numpy(full_state["seg_mask"]),
            "full_valid_mask": torch.from_numpy(full_state["valid_mask"]),
            "reg_head_ids": reg["head_ids"],
            "reg_head_time_state": torch.from_numpy(reg["time_state"]),
            "reg_head_source_indices": torch.from_numpy(reg["source_indices"]),
            "reg_head_target_indices": torch.from_numpy(reg["target_indices"]),
            "reg_head_params": torch.from_numpy(reg["params"]),
            "reg_head_params_raw": torch.from_numpy(reg["raw_params"]),
            "reg_head_mask": torch.from_numpy(reg["mask"]),
            "reg_head_mean_error": torch.from_numpy(reg["mean_error"]),
            "empty_view_centers": torch.from_numpy(empty_state["view_centers"]),
            "empty_target_center": torch.from_numpy(empty_state["target_center"]),
            "empty_target_scale": torch.from_numpy(empty_state["target_scale"]),
            "full_view_centers": torch.from_numpy(full_state["view_centers"]),
            "full_target_center": torch.from_numpy(full_state["target_center"]),
            "full_target_scale": torch.from_numpy(full_state["target_scale"]),
            "volume_mm3": torch.from_numpy(volume["volume_mm3"]),
            "volume_m3": torch.from_numpy(volume["volume_m3"]),
            "volume_mask": torch.from_numpy(volume["volume_mask"]),
            "volume_quality": torch.from_numpy(volume["volume_quality"]),
            "volume_geometry": torch.from_numpy(volume["volume_geometry"]),
        }
