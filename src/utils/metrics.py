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


def find_best_threshold_by_f1(
    y_true,
    y_prob,
    start: float = 0.05,
    end: float = 0.50,
    step: float = 0.01,
    min_recall: float = 0.30
):
    """
    Validation set üzerinde en iyi F1-score veren threshold değerini bulur.

    Anomali tespitinde çok yüksek threshold seçilirse model hiç anomali tahmin etmeyebilir.
    Bu nedenle threshold aralığı varsayılan olarak 0.05 - 0.50 arasında tutulmuştur.

    Ek olarak:
    - Önce recall değeri min_recall üzerinde olan threshold'lar tercih edilir.
    - F1 eşit veya çok yakınsa daha düşük threshold seçilir.
    """
    thresholds = np.arange(start, end + step, step)

    best_threshold = 0.5
    best_metrics = None
    best_f1 = -1

    fallback_threshold = 0.5
    fallback_metrics = None
    fallback_f1 = -1

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred
        )

        current_f1 = metrics["f1"]
        current_recall = metrics["recall"]

        # Fallback: recall şartı olmasa bile en iyi F1'i sakla
        if current_f1 > fallback_f1:
            fallback_f1 = current_f1
            fallback_threshold = float(threshold)
            fallback_metrics = metrics

        # Ana seçim: minimum recall şartını sağlayanlar arasından seç
        if current_recall >= min_recall:
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(threshold)
                best_metrics = metrics

    # Eğer hiçbir threshold min_recall şartını sağlamadıysa fallback kullan
    if best_metrics is None:
        best_threshold = fallback_threshold
        best_metrics = fallback_metrics

    return best_threshold, best_metrics