import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def calculate_tpr_fpr(y, y_pred, n_classes):
    tprs, fprs = [], []
    for i in range(n_classes):
        # 将类别 i 视为正例，其余类别为负例
        y_i = np.array(y[:, i])
        y_pred_i = np.array(y_pred[:, i])
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_i, y_pred_i, labels=[0, 1]).ravel()
        # 计算 TPR 和 FPR
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)
    return tprs, fprs


def cal_metrics(y, y_pred, n_classes):
    # 计算宏平均 TPR 和 FPR
    y_pred = np.eye(n_classes)[y_pred.argmax(axis=1)]
    tprs, fprs = calculate_tpr_fpr(y, y_pred, n_classes)
    macro_tpr = np.mean(tprs)
    macro_fpr = np.mean(fprs)
    # 计算微平均 TPR 和 FPR
    tn, fp, fn, tp = confusion_matrix(y.ravel(), y_pred.ravel()).ravel()
    micro_tpr = tp / (tp + fn)
    micro_fpr = fp / (fp + tn)
    # 计算其他指标
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y.argmax(axis=1), y_pred.argmax(axis=1), average="macro", zero_division=0)
    precision = precision_score(
        y.argmax(axis=1), y_pred.argmax(axis=1), average="macro", zero_division=0
    )
    recall = recall_score(y.argmax(axis=1), y_pred.argmax(axis=1), average="macro", zero_division=0)
    return {
        "macro_tpr": macro_tpr,
        "macro_fpr": macro_fpr,
        "micro_tpr": micro_tpr,
        "micro_fpr": micro_fpr,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
