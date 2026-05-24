from typing import List, Tuple, Optional


def calculate_levenshtein_distance(source: str, target: str) -> int:
    """
    İki string arasındaki Levenshtein edit distance değerini hesaplar.

    Edit distance:
    - karakter ekleme
    - karakter silme
    - karakter değiştirme

    işlemlerinden minimum kaç tanesiyle source stringinin target stringine
    dönüştürülebileceğini gösterir.
    """
    if source == target:
        return 0

    if len(source) == 0:
        return len(target)

    if len(target) == 0:
        return len(source)

    rows = len(source) + 1
    cols = len(target) + 1

    distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        distance_matrix[i][0] = i

    for j in range(cols):
        distance_matrix[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if source[i - 1] == target[j - 1]:
                substitution_cost = 0
            else:
                substitution_cost = 1

            distance_matrix[i][j] = min(
                distance_matrix[i - 1][j] + 1,
                distance_matrix[i][j - 1] + 1,
                distance_matrix[i - 1][j - 1] + substitution_cost
            )

    return distance_matrix[-1][-1]


def find_nearest_pattern(
    unseen_pattern: str,
    known_patterns: List[str]
) -> Tuple[Optional[str], Optional[int]]:
    """
    Eğitimde görülmeyen bir pattern için bilinen patternlar arasında
    Levenshtein distance değeri en küçük olanı bulur.

    Eşitlik durumunda alfabetik olarak önce gelen pattern seçilir.
    Bu, deterministik ve tekrar üretilebilir sonuç sağlar.
    """
    if not known_patterns:
        return None, None

    sorted_patterns = sorted(known_patterns)

    nearest_pattern = None
    nearest_distance = None

    for known_pattern in sorted_patterns:
        distance = calculate_levenshtein_distance(
            unseen_pattern,
            known_pattern
        )

        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_pattern = known_pattern

    return nearest_pattern, nearest_distance