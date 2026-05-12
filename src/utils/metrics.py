import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def calculate_classification_metrics(y_true, y_pred) -> dict:
    """
    Accuracy, precision, recall, F1-score ve confusion matrix hesaplar.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    return metrics


def find_best_threshold_by_f1(y_true, y_prob, start: float = 0.05, end: float = 0.95, step: float = 0.01):
    """
    Validation set üzerinde en iyi F1-score veren threshold değerini bulur.

    Not:
    Threshold test setine göre seçilmez.
    Test set yalnızca final değerlendirme için kullanılır.
    """
    thresholds = np.arange(start, end + step, step)

    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics