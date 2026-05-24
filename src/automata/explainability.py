from typing import Dict, List, Optional

from src.automata.transition_matrix import calculate_path_probability


def make_automata_decision(
    path_probability: float,
    anomaly_threshold: float
) -> str:
    """
    Path probability değerine göre karar üretir.

    Düşük olasılık:
    - Beklenmeyen davranış
    - Anomali adayı

    Yüksek olasılık:
    - Eğitimde öğrenilen normal davranışa daha yakın
    """
    if path_probability < anomaly_threshold:
        return "anomaly"

    return "normal"


def calculate_confidence_score(path_probability: float) -> float:
    """
    Güven skorunu path probability üzerinden hesaplar.

    Bu projede otomata kararının güven skoru,
    gözlemlenen pattern dizisinin olasılığı olarak ele alınmıştır.
    """
    return float(path_probability)


def explain_automata_decision(
    patterns: List[str],
    transition_probabilities: Dict[str, Dict[str, float]],
    anomaly_threshold: float,
    time_step: Optional[int] = None,
    observed_pattern: Optional[str] = None,
    status: str = "seen",
    mapped_to: Optional[str] = None,
    default_probability: float = 1e-12
) -> dict:
    """
    Olasılıksal otomata kararını açıklanabilir JSON formatında üretir.

    Üretilen açıklama:
    - state bilgisi
    - pattern bilgisi
    - unseen/seen durumu
    - varsa mapped pattern
    - transition detayları
    - path probability
    - karar
    - confidence score
    """
    path_probability, transitions = calculate_path_probability(
        patterns,
        transition_probabilities,
        default_probability=default_probability
    )

    decision = make_automata_decision(
        path_probability=path_probability,
        anomaly_threshold=anomaly_threshold
    )

    confidence_score = calculate_confidence_score(path_probability)

    current_state = patterns[-1] if patterns else None

    explanation = {
        "time_step": time_step,
        "state": current_state,
        "pattern": observed_pattern if observed_pattern is not None else current_state,
        "status": status,
        "mapped_to": mapped_to,
        "transitions": transitions,
        "probability": path_probability,
        "path_probability": path_probability,
        "anomaly_threshold": anomaly_threshold,
        "decision": decision,
        "confidence_score": confidence_score
    }

    return explanation