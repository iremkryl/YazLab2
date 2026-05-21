from src.automata.sax import apply_sax, create_sliding_window_patterns


def test_apply_sax_basic_example():
    """
    Basit SAX testi.

    Sayılar küçükten büyüğe gidiyor.
    alphabet_size = 3 olduğu için a, b, c sembolleri bekliyoruz.
    """

    paa_values = [1.0, 2.0, 3.0]

    symbols = apply_sax(paa_values, alphabet_size=3)

    assert symbols == ["a", "b", "c"]


def test_apply_sax_same_values():
    """
    Tüm değerler aynıysa hepsi ilk sembole atanmalı.
    """

    paa_values = [5.0, 5.0, 5.0]

    symbols = apply_sax(paa_values, alphabet_size=3)

    assert symbols == ["a", "a", "a"]


def test_sliding_window_patterns_basic_example():
    """
    Sliding window pattern testi.

    symbols = ["a", "b", "c", "c", "a"]
    window_size = 3

    Beklenen:
    ["abc", "bcc", "cca"]
    """

    symbols = ["a", "b", "c", "c", "a"]

    patterns = create_sliding_window_patterns(
        symbols=symbols,
        window_size=3
    )

    assert patterns == ["abc", "bcc", "cca"]


def test_sliding_window_invalid_window_size():
    """
    window_size sembol sayısından büyük olursa hata vermeli.
    """

    symbols = ["a", "b"]

    try:
        create_sliding_window_patterns(symbols, window_size=5)
        assert False
    except ValueError:
        assert True