from pathlib import Path
import numpy as np
from ml_from_scratch.classification.logistic_classifier import LogisticClassifier

root_path = Path.cwd().parents[1]
data_file = "lr_dataset.csv"

def test_logistic_definition():
    num_steps = 250
    learning_rate = 2e-3
    l2_coef = 1.0

