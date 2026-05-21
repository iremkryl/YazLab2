import numpy as np


def apply_paa(series, n_segments: int):
    """
    PAA (Piecewise Aggregate Approximation) uygular.

    Amaç:
    Uzun bir zaman serisini daha kısa hale getirmek.

    Örnek:
    series = [1, 2, 3, 4, 5, 6]
    n_segments = 3

    Çıktı:
    [1.5, 3.5, 5.5]

    Parametreler:
    - series: Tek boyutlu sayı dizisi
    - n_segments: Kaç parçaya bölüneceği

    Dönen değer:
    - Her parçanın ortalamasından oluşan NumPy array
    """

    series = np.asarray(series, dtype=float)

    if series.ndim != 1:
        raise ValueError("PAA yalnızca tek boyutlu veri ile çalışır.")

    if n_segments <= 0:
        raise ValueError("n_segments 0'dan büyük olmalıdır.")

    if n_segments > len(series):
        raise ValueError(
            "n_segments veri uzunluğundan büyük olamaz."
        )

    # Veriyi n_segments kadar parçaya bölüyoruz
    segments = np.array_split(series, n_segments)

    # Her parçanın ortalamasını alıyoruz
    paa_values = np.array([
        segment.mean()
        for segment in segments
    ])

    return paa_values