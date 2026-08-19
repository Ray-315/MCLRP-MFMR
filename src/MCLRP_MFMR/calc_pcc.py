from __future__ import annotations

import numpy as np


def _normalize_crossdata(cross_data, target_shape: tuple[int, int]) -> list[np.ndarray]:
    if isinstance(cross_data, list):
        iterator = cross_data
    elif isinstance(cross_data, np.ndarray):
        if cross_data.dtype == object:
            iterator = cross_data.flatten().tolist()
        elif cross_data.ndim >= 3:
            iterator = [cross_data[i] for i in range(cross_data.shape[0])]
        else:
            iterator = [cross_data]
    else:
        iterator = [cross_data]

    masks: list[np.ndarray] = []
    target_size = int(np.prod(target_shape))
    for idx, mask in enumerate(iterator, start=1):
        arr = np.asarray(mask)
        if arr.shape != target_shape and arr.size == target_size:
            arr = arr.reshape(target_shape)
        if arr.size != target_size:
            raise ValueError(
                f"Mask size mismatch at fold {idx}: mask size {arr.size}, "
                f"shape {arr.shape}, target shape {target_shape}"
            )
        masks.append(arr)
    return masks


def calc_pcc(w: np.ndarray, MM: np.ndarray, crossData):
    cross_list = _normalize_crossdata(crossData, MM.shape)
    fold_pcc = np.zeros(len(cross_list), dtype=float)
    mask_all = np.zeros(MM.size, dtype=bool)
    y_true_flat = np.ravel(MM)
    y_pred_flat = np.ravel(w)

    for idx, mask in enumerate(cross_list):
        mask_bool = np.ravel(mask != 0)
        mask_all |= mask_bool
        y_true = y_true_flat[mask_bool]
        y_pred = y_pred_flat[mask_bool]
        if y_pred.size < 2 or np.std(y_pred) == 0 or np.std(y_true) == 0:
            fold_pcc[idx] = 0.0
        else:
            value = float(np.corrcoef(y_true, y_pred)[0, 1])
            fold_pcc[idx] = value if np.isfinite(value) else 0.0

    y_true_all = y_true_flat[mask_all]
    y_pred_all = y_pred_flat[mask_all]
    if y_pred_all.size < 2 or np.std(y_pred_all) == 0 or np.std(y_true_all) == 0:
        overall_pcc = 0.0
    else:
        value = float(np.corrcoef(y_true_all, y_pred_all)[0, 1])
        overall_pcc = value if np.isfinite(value) else 0.0
    return overall_pcc, fold_pcc
