from src.automata.transition_matrix import (
    build_transition_counts,
    build_transition_probabilities,
    calculate_path_probability
)


def test_build_transition_counts():
    patterns = ["aaa", "aab", "aaa", "aab", "abb"]

    counts = build_transition_counts(patterns)

    assert counts["aaa"]["aab"] == 2
    assert counts["aab"]["aaa"] == 1
    assert counts["aab"]["abb"] == 1


def test_build_transition_probabilities():
    patterns = ["aaa", "aab", "aaa", "aab", "abb"]

    probabilities = build_transition_probabilities(patterns)

    assert probabilities["aaa"]["aab"] == 1.0
    assert probabilities["aab"]["aaa"] == 0.5
    assert probabilities["aab"]["abb"] == 0.5


def test_calculate_path_probability():
    patterns = ["aaa", "aab", "abb"]

    transition_probabilities = {
        "aaa": {
            "aab": 1.0
        },
        "aab": {
            "abb": 0.5
        }
    }

    path_probability, transitions = calculate_path_probability(
        patterns,
        transition_probabilities
    )

    assert path_probability == 0.5
    assert len(transitions) == 2
    assert transitions[0]["from_state"] == "aaa"
    assert transitions[0]["to_state"] == "aab"
    assert transitions[0]["probability"] == 1.0