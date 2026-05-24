from src.automata.levenshtein import (
    calculate_levenshtein_distance,
    find_nearest_pattern
)


def test_levenshtein_distance_same_strings():
    distance = calculate_levenshtein_distance("abc", "abc")

    assert distance == 0


def test_levenshtein_distance_one_substitution():
    distance = calculate_levenshtein_distance("abc", "adc")

    assert distance == 1


def test_levenshtein_distance_insertion():
    distance = calculate_levenshtein_distance("abc", "abdc")

    assert distance == 1


def test_find_nearest_pattern():
    known_patterns = ["aaa", "abc", "bbb"]

    nearest_pattern, distance = find_nearest_pattern(
        unseen_pattern="adc",
        known_patterns=known_patterns
    )

    assert nearest_pattern == "abc"
    assert distance == 1