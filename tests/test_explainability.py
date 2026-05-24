from src.automata.explainability import (
    make_automata_decision,
    explain_automata_decision
)


def test_make_automata_decision_anomaly():
    decision = make_automata_decision(
        path_probability=0.10,
        anomaly_threshold=0.20
    )

    assert decision == "anomaly"


def test_make_automata_decision_normal():
    decision = make_automata_decision(
        path_probability=0.50,
        anomaly_threshold=0.20
    )

    assert decision == "normal"


def test_explain_automata_decision():
    patterns = ["aaa", "aab", "abb"]

    transition_probabilities = {
        "aaa": {
            "aab": 1.0
        },
        "aab": {
            "abb": 0.5
        }
    }

    explanation = explain_automata_decision(
        patterns=patterns,
        transition_probabilities=transition_probabilities,
        anomaly_threshold=0.60,
        time_step=10,
        observed_pattern="abb",
        status="seen",
        mapped_to=None
    )

    assert explanation["time_step"] == 10
    assert explanation["state"] == "abb"
    assert explanation["pattern"] == "abb"
    assert explanation["status"] == "seen"
    assert explanation["mapped_to"] is None
    assert explanation["path_probability"] == 0.5
    assert explanation["decision"] == "anomaly"
    assert explanation["confidence_score"] == 0.5
    assert len(explanation["transitions"]) == 2