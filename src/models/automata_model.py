from typing import Dict, List, Optional

from src.automata.transition_matrix import (
    build_transition_probabilities,
    calculate_path_probability
)
from src.automata.explainability import explain_automata_decision
from src.automata.levenshtein import find_nearest_pattern


class ProbabilisticAutomataModel:
    """
    Olasılıksal otomata modeli.

    Bu model:
    1. Eğitim pattern dizisinden state geçiş olasılıklarını öğrenir.
    2. Test pattern dizisinin path probability değerini hesaplar.
    3. Düşük olasılıklı dizileri anomali olarak işaretler.
    4. Kararını açıklanabilir JSON formatında döndürür.
    """

    def __init__(
        self,
        anomaly_threshold: float = 1e-6,
        smoothing: float = 0.0,
        default_probability: float = 1e-12
    ):
        self.anomaly_threshold = anomaly_threshold
        self.smoothing = smoothing
        self.default_probability = default_probability

        self.transition_probabilities: Dict[str, Dict[str, float]] = {}
        self.train_patterns: List[str] = []
        self.states: set = set()
        self.is_fitted = False

    def fit(self, train_patterns: List[str]):
        """
        Eğitim pattern dizisi üzerinden transition probability tablosunu oluşturur.
        """
        if not train_patterns:
            raise ValueError("train_patterns boş olamaz.")

        if len(train_patterns) < 2:
            raise ValueError("Transition hesaplamak için en az 2 pattern gereklidir.")

        self.train_patterns = list(train_patterns)
        self.states = set(train_patterns)

        self.transition_probabilities = build_transition_probabilities(
            patterns=train_patterns,
            smoothing=self.smoothing
        )

        self.is_fitted = True

        return self

    def _check_is_fitted(self):
        """
        Model eğitilmeden predict çağrılmasını engeller.
        """
        if not self.is_fitted:
            raise RuntimeError("Model henüz fit edilmedi. Önce fit() çağrılmalıdır.")

    def detect_unseen_patterns(self, patterns: List[str]) -> List[str]:
        """
        Eğitimde görülmeyen pattern'ları tespit eder.
        """
        self._check_is_fitted()

        unseen_patterns = [
            pattern for pattern in patterns
            if pattern not in self.states
        ]

        return unseen_patterns

    def calculate_sequence_probability(self, patterns: List[str]) -> float:
        """
        Verilen pattern dizisinin path probability değerini hesaplar.
        """
        self._check_is_fitted()

        path_probability, _ = calculate_path_probability(
            patterns=patterns,
            transition_probabilities=self.transition_probabilities,
            default_probability=self.default_probability
        )

        return path_probability

    def predict_sequence(
        self,
        patterns: List[str],
        time_step: Optional[int] = None
    ) -> dict:
        """
        Tek bir pattern dizisi için karar ve açıklama üretir.

        Eğer test sırasında eğitimde görülmeyen pattern varsa,
        Levenshtein distance ile eğitim sözlüğündeki en yakın pattern'a eşlenir.
        """
        self._check_is_fitted()

        if not patterns:
            raise ValueError("patterns boş olamaz.")

        unseen_patterns = self.detect_unseen_patterns(patterns)

        mapped_patterns = list(patterns)
        status = "seen"
        observed_pattern = patterns[-1]
        mapped_to = None
        levenshtein_distance = None

        if unseen_patterns:
            status = "unseen"
            observed_pattern = unseen_patterns[0]

            nearest_pattern, distance = find_nearest_pattern(
                unseen_pattern=observed_pattern,
                known_patterns=list(self.states)
            )

            mapped_to = nearest_pattern
            levenshtein_distance = distance

            mapped_patterns = [
                mapped_to if pattern == observed_pattern else pattern
                for pattern in patterns
            ]

        explanation = explain_automata_decision(
            patterns=mapped_patterns,
            transition_probabilities=self.transition_probabilities,
            anomaly_threshold=self.anomaly_threshold,
            time_step=time_step,
            observed_pattern=observed_pattern,
            status=status,
            mapped_to=mapped_to,
            default_probability=self.default_probability
        )

        explanation["original_patterns"] = patterns
        explanation["used_patterns"] = mapped_patterns
        explanation["levenshtein_distance"] = levenshtein_distance

        return explanation

    def predict(self, pattern_sequences: List[List[str]]) -> List[dict]:
        """
        Birden fazla pattern dizisi için karar üretir.
        """
        self._check_is_fitted()

        explanations = []

        for index, patterns in enumerate(pattern_sequences):
            explanation = self.predict_sequence(
                patterns=patterns,
                time_step=index
            )

            explanations.append(explanation)

        return explanations

    def get_transition_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Öğrenilen transition probability tablosunu döndürür.
        """
        self._check_is_fitted()

        return self.transition_probabilities