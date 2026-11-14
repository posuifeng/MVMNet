from __future__ import annotations
import numpy as np

Z_SOURCE_MAX = 42000.0
Z_TARGET_MAX = 7000.0
Z_SCALE = Z_TARGET_MAX / Z_SOURCE_MAX


def rescale_points_z(points: np.ndarray) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    if out.ndim != 2 or out.shape[1] < 3:
        raise ValueError(f"Expected point cloud shape (N, >=3), got {out.shape}")
    out[:, 2] = np.clip(out[:, 2], 0.0, Z_SOURCE_MAX) * Z_SCALE
    return out


def rescale_special_params_z(params: np.ndarray) -> np.ndarray:
    out = np.asarray(params, dtype=np.float32).copy()
    if out.shape[-1] != 6:
        raise ValueError(f"Expected special params shape (6,), got {out.shape}")
    out[1] *= Z_SCALE
    out[5] *= Z_SCALE
    return out


def normalize_pair(
    source: np.ndarray, target: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    source_shifted = source - source_center
    target_shifted = target - target_center
    scale = (
        np.linalg.norm(target_shifted.max(axis=0) - target_shifted.min(axis=0)) + 1e-06
    )
    source_norm = source_shifted / scale
    target_norm = target_shifted / scale
    a, b, c, tx, ty, tz = params.astype(np.float32)
    matrix = np.array([[a, 0.0, 0.0], [0.0, 1.0, 0.0], [b, 0.0, c]], dtype=np.float32)
    t = np.array([tx, ty, tz], dtype=np.float32)
    source_center_vec = source_center.reshape(3).astype(np.float32)
    target_center_vec = target_center.reshape(3).astype(np.float32)
    t_norm = (matrix @ source_center_vec + t - target_center_vec) / scale
    params_norm = np.array([a, b, c, t_norm[0], t_norm[1], t_norm[2]], dtype=np.float32)
    meta = np.array([scale], dtype=np.float32)
    centers = np.stack([source_center_vec, target_center_vec], axis=0).astype(
        np.float32
    )
    return (source_norm, target_norm, params_norm, meta, centers)


def normalize_pair_for_inference(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    source_shifted = source - source_center
    target_shifted = target - target_center
    scale = (
        np.linalg.norm(target_shifted.max(axis=0) - target_shifted.min(axis=0)) + 1e-06
    )
    source_norm = source_shifted / scale
    target_norm = target_shifted / scale
    meta = np.array([scale], dtype=np.float32)
    centers = np.stack(
        [source_center.reshape(3), target_center.reshape(3)], axis=0
    ).astype(np.float32)
    return (
        source_norm.astype(np.float32),
        target_norm.astype(np.float32),
        meta,
        centers,
    )
