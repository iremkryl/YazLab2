from collections import defaultdict
from typing import Dict, List, Tuple


def build_transition_counts(patterns: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Pattern/state dizisinden geçiş sayılarını hesaplar.

    Örnek:
    patterns = ["aaa", "aab", "abb"]

    Geçişler:
    aaa -> aab
    aab -> abb
    """
    transition_counts = defaultdict(lambda: defaultdict(int))

    if len(patterns) < 2:
        return {}

    for i in range(len(patterns) - 1):
        current_state = patterns[i]
        next_state = patterns[i + 1]

        transition_counts[current_state][next_state] += 1

    return {
        state: dict(next_states)
        for state, next_states in transition_counts.items()
    }


def build_transition_probabilities(
    patterns: List[str],
    smoothing: float = 0.0
) -> Dict[str, Dict[str, float]]:
    """
    Pattern/state dizisinden geçiş olasılıklarını hesaplar.

    P(Si -> Sj) = Geçiş Sayısı / Toplam Çıkış Sayısı

    smoothing:
    - 0.0 ise sadece gözlenen geçişler kullanılır.
    - 0'dan büyük verilirse tüm state'ler için küçük olasılık eklenir.
    """
    transition_counts = build_transition_counts(patterns)

    if len(patterns) < 2:
        return {}

    unique_states = sorted(set(patterns))
    transition_probabilities = {}

    for current_state, next_counts in transition_counts.items():
        total_outgoing = sum(next_counts.values())

        transition_probabilities[current_state] = {}

        if smoothing > 0:
            denominator = total_outgoing + smoothing * len(unique_states)

            for next_state in unique_states:
                count = next_counts.get(next_state, 0)
                probability = (count + smoothing) / denominator
                transition_probabilities[current_state][next_state] = probability

        else:
            for next_state, count in next_counts.items():
                probability = count / total_outgoing
                transition_probabilities[current_state][next_state] = probability

    return transition_probabilities


def get_transition_probability(
    transition_probabilities: Dict[str, Dict[str, float]],
    current_state: str,
    next_state: str,
    default_probability: float = 1e-12
) -> float:
    """
    İki state arasındaki geçiş olasılığını döndürür.

    Eğer geçiş eğitimde hiç görülmediyse default_probability döner.
    Bu sayede path probability sıfıra düşmeden hesaplanabilir.
    """
    return transition_probabilities.get(
        current_state, {}
    ).get(
        next_state,
        default_probability
    )


def calculate_path_probability(
    patterns: List[str],
    transition_probabilities: Dict[str, Dict[str, float]],
    default_probability: float = 1e-12
) -> Tuple[float, List[dict]]:
    """
    Bir pattern/state dizisinin toplam path probability değerini hesaplar.

    P(sequence) = P(S1 -> S2) * P(S2 -> S3) * ...

    Ayrıca her geçişin detayını da döndürür.
    """
    if len(patterns) < 2:
        return 1.0, []

    path_probability = 1.0
    transition_details = []

    for i in range(len(patterns) - 1):
        current_state = patterns[i]
        next_state = patterns[i + 1]

        probability = get_transition_probability(
            transition_probabilities,
            current_state,
            next_state,
            default_probability=default_probability
        )

        path_probability *= probability

        transition_details.append({
            "from_state": current_state,
            "to_state": next_state,
            "probability": probability
        })

    return path_probability, transition_details