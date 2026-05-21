import numpy as np


def apply_sax(paa_values, alphabet_size: int = 3):
    """
    SAX (Symbolic Aggregate approXimation) uygular.

    Amaç:
    PAA'dan gelen sayısal değerleri sembollere/harflere çevirmek.

    Örnek:
    paa_values = [1.0, 2.0, 3.0]
    alphabet_size = 3

    Çıktı:
    ["a", "b", "c"]

    Parametreler:
    - paa_values: PAA'dan çıkan tek boyutlu sayı dizisi
    - alphabet_size: Kaç farklı sembol kullanılacağı

    Dönen değer:
    - Sembollerden oluşan liste
    """

    paa_values = np.asarray(paa_values, dtype=float)

    if paa_values.ndim != 1:
        raise ValueError("SAX yalnızca tek boyutlu veri ile çalışır.")

    if alphabet_size < 2:
        raise ValueError("alphabet_size en az 2 olmalıdır.")

    if alphabet_size > 26:
        raise ValueError("alphabet_size en fazla 26 olabilir.")

    alphabet = [
        chr(ord("a") + i)
        for i in range(alphabet_size)
    ]

    min_value = np.min(paa_values)
    max_value = np.max(paa_values)

    if min_value == max_value:
        return [alphabet[0] for _ in paa_values]

    bins = np.linspace(
        min_value,
        max_value,
        alphabet_size + 1
    )

    symbols = []

    for value in paa_values:
        bin_index = np.digitize(value, bins[1:-1], right=True)
        symbols.append(alphabet[bin_index])

    return symbols


def create_sliding_window_patterns(symbols, window_size: int):
    """
    Sembollerden sliding window ile pattern çıkarır.

    Örnek:
    symbols = ["a", "b", "c", "c", "a"]
    window_size = 3

    Çıktı:
    ["abc", "bcc", "cca"]

    Parametreler:
    - symbols: SAX sonucunda oluşan sembol listesi
    - window_size: Bir pattern kaç sembolden oluşacak

    Dönen değer:
    - Pattern listesi
    """

    if window_size <= 0:
        raise ValueError("window_size 0'dan büyük olmalıdır.")

    if window_size > len(symbols):
        raise ValueError("window_size sembol sayısından büyük olamaz.")

    patterns = []

    for i in range(len(symbols) - window_size + 1):
        pattern = "".join(symbols[i:i + window_size])
        patterns.append(pattern)

    return patterns