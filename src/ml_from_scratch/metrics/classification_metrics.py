import numpy as np
"""
这个模块负责保存分类模型相关的评价方法
"""

def acc(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """
    计算模型预测的准确率，返回准确率
    """
    return np.mean(y_true == y_pred)

def auc(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """
    计算不同阈值下模型预测的TP与FP，返回ROC曲线的AUC面积
    """
    # 按照预测概率从大到小排序，越靠前的样本预测为正类的概率越大，预测为负类的概率越小
    ## 阈值设定为各个不同的值，通过累计相加计算阈值调整时，相应的TP与FP值与比率
    idx = np.argsort(y_pred, axis=0)[::-1]
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1-y_true)
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    ## 利用积分定义计算AUC的面积值
    auc = 0
    for i in range(0,len(tpr)-1):
        auc += (fpr[i+1] - fpr[i])*tpr[i+1]

    return auc



