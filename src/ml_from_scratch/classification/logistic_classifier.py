import numpy as np
from ..metrics import classification_metrics as metrics

class LogisticClassifier:
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    _learning_rate: float
    _l2_coef: float
    theta: np.ndarray

    def __init__(self, lr, l2_coef):
        self._learning_rate = lr
        self._l2_coef = l2_coef

    def load_train_and_valid(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.theta = np.random.normal(size=X_train.shape[1])

    def load_test(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    @property
    def l2_coef(self):
        return self._l2_coef

    @property
    def learning_rate(self):
        return self._learning_rate

    @staticmethod
    def logistic(z: np.ndarray):
        return 1 / (1 + np.exp(-z))

    def GD(self, num_steps=100):
        """
        Returns
        -------
        train_losses, valid_losses, train_acc, valid_acc, train_auc, valid_auc
        """
        train_losses = []
        valid_losses = []
        train_acc = []
        valid_acc = []
        train_auc = []
        valid_auc = []

        for i in range(num_steps):
            pred = self.logistic(self.X_train @ self.theta)
            grad = -self.X_train.T @ (self.y_train - pred) + self.l2_coef * self.theta
            # 记录损失函数
            train_loss = - (self.y_train.T @ np.log(pred)) - ((1-self.y_train).T @ np.log(1-pred)) + (self.l2_coef/2)*(np.linalg.norm(self.theta)**2)
            train_losses.append(train_loss / len(self.X_train))
            valid_pred = self.logistic(self.X_valid @ self.theta)
            valid_loss = - (self.y_valid @ np.log(valid_pred)) - ((1-self.y_valid).T @ np.log(1-valid_pred))
            valid_losses.append(valid_loss / len(self.X_valid))
            # 记录各类评价指标，阈值采取0.5
            train_acc.append(metrics.acc(y_true=self.y_train, y_pred=(pred>=0.5)))
            valid_acc.append(metrics.acc(y_true=self.y_valid, y_pred=(valid_pred>=0.5)))
            train_auc.append(metrics.auc(y_true=self.y_train, y_pred=(pred>=0.5)))
            valid_auc.append(metrics.auc(y_true=self.y_valid, y_pred=(valid_pred>=0.5)))
            # 更新梯度，进入下一轮训练
            self.theta -= self.learning_rate * grad

        return train_losses, valid_losses, train_acc, valid_acc, train_auc, valid_auc

