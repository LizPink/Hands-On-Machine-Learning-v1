"""
该测试Module负责检测data_processor部分的功能
"""
import numpy as np
import ml_from_scratch.data.data_processor as processor
import pytest

rng = np.random.default_rng(seed=42)
split_ratio = 0.7

def test_load_csv():
    from pathlib import Path
    root_path = Path.cwd()
    data_file = Path("data/raw/logistic_classification/lr_dataset.csv")

    data = processor.load_csv(path=root_path/data_file)
    assert isinstance(data, np.ndarray)
    assert data.shape == (1000, 3)


def test_train_test_split():
    ...