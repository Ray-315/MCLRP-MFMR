"""
数据加载工具：读取由 .mat 转换来的 .npz（allow_pickle=True）。
"""
from typing import Any, Union
from pathlib import Path
import numpy as np


def load_npz_var(path: Union[str, Path], var: str = None) -> Any:
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)
    if var and var in data:
        return data[var]
    # 如果未指定变量名，则取第一个键
    key = list(data.keys())[0]
    return data[key]
