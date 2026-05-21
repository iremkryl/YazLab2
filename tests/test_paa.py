import numpy as np

from src.automata.paa import apply_paa


def test_paa_basic_example():
    """
    Basit PAA testi.

    [1, 2, 3, 4, 5, 6] listesini 3 parçaya bölersek:

    [1, 2] -> 1.5
    [3, 4] -> 3.5
    [5, 6] -> 5.5
    """

    series = [1, 2, 3, 4, 5, 6]

    result = apply_paa(series, n_segments=3)

    expected = np.array([1.5, 3.5, 5.5])

    assert np.allclose(result, expected)


def test_paa_invalid_segment_count():
    """
    n_segments veri uzunluğundan büyük olursa hata vermeli.
    """

    series = [1, 2, 3]

    try:
        apply_paa(series, n_segments=5)
        assert False
    except ValueError:
        assert True