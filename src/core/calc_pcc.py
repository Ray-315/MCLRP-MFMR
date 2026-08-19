import numpy as np


def _normalize_crossdata(crossData, target_shape):
    """Return list of fold masks as ndarrays matching target_shape (or same size)."""
    cross_list = []
    if isinstance(crossData, list):
        iter_obj = crossData
    elif isinstance(crossData, np.ndarray):
        if crossData.dtype == object:
            iter_obj = crossData.flatten().tolist()
        elif crossData.ndim >= 3:
            iter_obj = [crossData[i] for i in range(crossData.shape[0])]
        else:
            iter_obj = [crossData]
    else:
        iter_obj = [crossData]

    for idx, m in enumerate(iter_obj, start=1):
        mask_arr = np.array(m)
        if mask_arr.shape != target_shape and mask_arr.size == np.prod(target_shape):
            mask_arr = mask_arr.reshape(target_shape)
        if mask_arr.size != np.prod(target_shape):
            raise ValueError(
                f"Mask size mismatch at fold {idx}: mask size {mask_arr.size}, shape {mask_arr.shape}, "
                f"target size {np.prod(target_shape)}, target shape {target_shape}"
            )
        cross_list.append(mask_arr)
    return cross_list


def calc_pcc(w: np.ndarray, MM: np.ndarray, crossData):
    """
    对应 MATLAB calc_pcc.m
    crossData: list/array of fold matrices（与 w、MM 同形状）
    返回 overallPCC, foldPCC
    """
    cross_list = _normalize_crossdata(crossData, MM.shape)

    numFolds = len(cross_list)
    foldPCC = np.zeros(numFolds, dtype=float)
    maskAll = np.zeros(MM.size, dtype=bool)

    MM_flat = np.ravel(MM)
    w_flat = np.ravel(w)

    for i, mask in enumerate(cross_list):
        mask_bool = np.ravel(mask != 0)
        maskAll |= mask_bool
        yTrue = MM_flat[mask_bool]
        yPred = w_flat[mask_bool]
        if yPred.size < 2 or np.std(yPred) == 0 or np.std(yTrue) == 0:
            foldPCC[i] = 0.0
            continue
        c = np.corrcoef(yTrue, yPred)
        foldPCC[i] = c[0, 1]

    yTrueAll = MM_flat[maskAll]
    yPredAll = w_flat[maskAll]
    if yPredAll.size < 2 or np.std(yPredAll) == 0 or np.std(yTrueAll) == 0:
        overallPCC = 0.0
    else:
        overallPCC = np.corrcoef(yTrueAll, yPredAll)[0, 1]

    return overallPCC, foldPCC
