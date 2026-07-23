import numpy as np
import ml_from_scratch.metrics.classification_metrics as metrics


def test_accuracy_all_correct():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    result = metrics.acc(y_true, y_pred)

    assert result == 1.0


def test_accuracy_partially_correct():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1])

    result = metrics.acc(y_true, y_pred)

    assert result == 0.5


def test_accuracy_all_wrong():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1])

    result = metrics.acc(y_true, y_pred)

    assert result == 0.0