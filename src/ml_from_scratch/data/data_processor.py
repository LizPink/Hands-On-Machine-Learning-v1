"""
该Module存放了一些用于数据处理的工具
"""
import numpy as np
from pathlib import Path
from typing import TypeAlias

# 类型定义
SplitResult: TypeAlias = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def load_csv(path:Path) -> np.ndarray:
    data = np.loadtxt(fname=path, delimiter=",")
    return data

def train_test_split(X:np.ndarray, y:np.ndarray, train_ratio:float=0.7, seed:int=42) -> SplitResult:
    rng = np.random.default_rng(seed=seed)
    idx = rng.permutation(len(X))
    split_num = int(len(X)*train_ratio)
    # 随机打乱数据
    X = X[idx]
    y = y[idx]
    # 划分训练数据与验证/测试数据
    X_train = X[0:split_num]
    X_test = X[split_num:]
    y_train = y[0:split_num]
    y_test = y[split_num:]
    # 返回数据划分结果
    return X_train, X_test, y_train, y_test

def cat_constant(X:np.ndarray):
    X = np.concatenate([np.ones(shape=(X.shape[0],1)),X], axis=1)
    return X
