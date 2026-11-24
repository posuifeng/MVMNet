from __future__ import annotations
import numpy as np


def apply_transform(
    points: np.ndarray, matrix: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return points @ matrix.T + translation


def special_matrix(a: float, b: float, c: float) -> np.ndarray:
    return np.array([[a, 0.0, 0.0], [0.0, 1.0, 0.0], [b, 0.0, c]], dtype=np.float64)


def special_params_to_matrix_translation(
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(params, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Expected params shape (6,), got {values.shape}")
    a, b, c, tx, ty, tz = values
    matrix = special_matrix(float(a), float(b), float(c))
    translation = np.array([tx, ty, tz], dtype=np.float64)
    return (matrix, translation)
