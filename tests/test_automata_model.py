from src.models.automata_model import ProbabilisticAutomataModel


def test_automata_model_fit_creates_transition_probabilities():
    train_patterns = ["aaa", "aab", "abb", "aaa", "aab", "abb"]

    model = ProbabilisticAutomataModel()
    model.fit(train_patterns)

    probabilities = model.get_transition_probabilities()

    assert "aaa" in probabilities
    assert probabilities["aaa"]["aab"] == 1.0
    assert probabilities["aab"]["abb"] == 1.0


def test_automata_model_predict_unseen_sequence_with_levenshtein_mapping():
    train_patterns = ["aaa", "abc", "bbb", "aaa", "abc", "bbb"]

    model = ProbabilisticAutomataModel(
        anomaly_threshold=0.5,
        default_probability=1e-12
    )

    model.fit(train_patterns)

    explanation = model.predict_sequence(
        patterns=["aaa", "adc"],
        time_step=10
    )

    assert explanation["time_step"] == 10
    assert explanation["status"] == "unseen"
    assert explanation["pattern"] == "adc"
    assert explanation["mapped_to"] == "abc"
    assert explanation["levenshtein_distance"] == 1
    assert explanation["original_patterns"] == ["aaa", "adc"]
    assert explanation["used_patterns"] == ["aaa", "abc"]


def test_automata_model_predict_unseen_sequence_as_anomaly():
    train_patterns = ["aaa", "aab", "abb", "aaa", "aab", "abb"]

    model = ProbabilisticAutomataModel(
        anomaly_threshold=0.5,
        default_probability=1e-12
    )

    model.fit(train_patterns)

    explanation = model.predict_sequence(
        patterns=["aaa", "xyz"],
        time_step=10
    )

    assert explanation["time_step"] == 10
    assert explanation["status"] == "unseen"
    assert explanation["pattern"] == "xyz"
    assert explanation["decision"] == "anomaly"
    assert explanation["path_probability"] == 1e-12