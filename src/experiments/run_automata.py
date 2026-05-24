import numpy as np

from src.config.config_loader import load_config
from src.data.load_batadal import (
    load_batadal,
    split_batadal_time_ordered,
    prepare_batadal_features_and_target
)
from src.preprocessing.scaler import fit_transform_scaler, transform_with_scaler
from src.preprocessing.pca_transformer import fit_transform_pca, transform_with_pca
from src.automata.paa import apply_paa
from src.automata.sax import apply_sax, create_sliding_window_patterns
from src.models.automata_model import ProbabilisticAutomataModel
from src.utils.metrics import calculate_classification_metrics
from src.utils.logger import save_json_result


def prepare_batadal_automata_data(config: dict):
    """
    BATADAL verisini otomata modeline uygun hale getirir.

    Otomata modeli tek boyutlu veriyle çalıştığı için:
    1. Veri okunur.
    2. Zaman sıralı train/validation/test ayrımı yapılır.
    3. X/y ayrılır.
    4. Normalizasyon sadece train üzerinde fit edilir.
    5. PCA sadece train üzerinde fit edilir.
    6. PC1 değerleri döndürülür.
    """
    batadal_config = config["datasets"]["batadal"]
    preprocessing_config = config["preprocessing"]

    df = load_batadal(batadal_config["raw_path"])

    train_df, validation_df, test_df = split_batadal_time_ordered(
        df,
        train_ratio=batadal_config["train_ratio"],
        validation_ratio=batadal_config["validation_ratio"],
        test_ratio=batadal_config["test_ratio"]
    )

    X_train, y_train = prepare_batadal_features_and_target(
        train_df,
        target_column=batadal_config["target_column"],
        time_column=batadal_config["time_column"],
        normal_label=batadal_config["normal_label"],
        anomaly_label=batadal_config["anomaly_label"]
    )

    X_val, y_val = prepare_batadal_features_and_target(
        validation_df,
        target_column=batadal_config["target_column"],
        time_column=batadal_config["time_column"],
        normal_label=batadal_config["normal_label"],
        anomaly_label=batadal_config["anomaly_label"]
    )

    X_test, y_test = prepare_batadal_features_and_target(
        test_df,
        target_column=batadal_config["target_column"],
        time_column=batadal_config["time_column"],
        normal_label=batadal_config["normal_label"],
        anomaly_label=batadal_config["anomaly_label"]
    )

    X_train_scaled, scaler = fit_transform_scaler(
        X_train,
        method=preprocessing_config["normalization"]
    )

    X_val_scaled = transform_with_scaler(X_val, scaler)
    X_test_scaled = transform_with_scaler(X_test, scaler)

    X_train_pca, pca = fit_transform_pca(
        X_train_scaled,
        n_components=preprocessing_config["pca_components"]
    )

    X_val_pca = transform_with_pca(X_val_scaled, pca)
    X_test_pca = transform_with_pca(X_test_scaled, pca)

    train_pc1 = X_train_pca.flatten()
    val_pc1 = X_val_pca.flatten()
    test_pc1 = X_test_pca.flatten()

    return train_pc1, y_train, val_pc1, y_val, test_pc1, y_test


def convert_series_to_patterns(
    series,
    n_segments: int,
    alphabet_size: int,
    window_size: int
):
    """
    Tek boyutlu zaman serisini otomata pattern listesine dönüştürür.

    Akış:
    PC1 seri -> PAA -> SAX -> sliding window pattern
    """
    paa_values = apply_paa(
        series=series,
        n_segments=n_segments
    )

    symbols = apply_sax(
        paa_values=paa_values,
        alphabet_size=alphabet_size
    )

    patterns = create_sliding_window_patterns(
        symbols=symbols,
        window_size=window_size
    )

    return patterns, paa_values, symbols


def create_pattern_windows(patterns, sequence_size: int = 3):
    """
    Pattern listesinden otomata modelinin değerlendireceği küçük pattern dizileri üretir.

    Örnek:
    patterns = ["aaa", "aab", "abb", "bbb"]
    sequence_size = 3

    Çıktı:
    [
        ["aaa", "aab", "abb"],
        ["aab", "abb", "bbb"]
    ]
    """
    if sequence_size <= 1:
        raise ValueError("sequence_size en az 2 olmalıdır.")

    if sequence_size > len(patterns):
        raise ValueError("sequence_size pattern sayısından büyük olamaz.")

    pattern_windows = []

    for i in range(len(patterns) - sequence_size + 1):
        window = patterns[i:i + sequence_size]
        pattern_windows.append(window)

    return pattern_windows


def align_labels_to_pattern_windows(y, n_segments: int, window_size: int, sequence_size: int):
    """
    Orijinal y etiketlerini pattern window sayısıyla hizalar.

    PAA sonrası zaman serisi n_segments uzunluğa iner.
    Bu yüzden orijinal y değerleri de n_segments parçaya bölünür.
    Her segment için, o segmente denk gelen orijinal etiketlerde anomali varsa 1 kabul edilir.

    Ardından sliding window ve sequence window kayıpları hesaba katılır.
    """
    y_values = np.asarray(y)

    if n_segments > len(y_values):
        raise ValueError("n_segments y uzunluğundan büyük olamaz.")

    segments = np.array_split(y_values, n_segments)

    segment_labels = np.array([
        1 if np.any(segment == 1) else 0
        for segment in segments
    ])

    # SAX sembolleri n_segments uzunluğunda olur.
    # Sliding window sonrası pattern sayısı:
    # n_segments - window_size + 1
    pattern_labels = segment_labels[window_size - 1:]

    # Pattern sequence sonrası örnek sayısı:
    # pattern_count - sequence_size + 1
    final_labels = pattern_labels[sequence_size - 1:]

    return final_labels


def find_best_automata_threshold(
    model: ProbabilisticAutomataModel,
    validation_pattern_windows,
    y_val_aligned
):
    """
    Validation set üzerinde en iyi F1-score veren anomaly threshold değerini bulur.

    Düşük path probability anomaly kabul edildiği için:
    path_probability < threshold -> anomaly
    """
    probabilities = []

    for pattern_window in validation_pattern_windows:
        probability = model.calculate_sequence_probability(pattern_window)
        probabilities.append(probability)

    probabilities = np.array(probabilities)

    candidate_thresholds = np.unique(probabilities)

    if len(candidate_thresholds) == 0:
        raise ValueError("Validation probability listesi boş.")

    best_threshold = float(candidate_thresholds[0])
    best_metrics = None
    best_f1 = -1

    for threshold in candidate_thresholds:
        y_pred = (probabilities < threshold).astype(int)

        metrics = calculate_classification_metrics(
            y_true=y_val_aligned,
            y_pred=y_pred
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics, probabilities


def run_batadal_automata_experiment():
    """
    BATADAL üzerinde olasılıksal otomata deneyini çalıştırır.
    """
    config = load_config()

    automata_config = config["automata"]
    outputs_config = config["outputs"]

    window_size = automata_config["default_window_size"]
    alphabet_size = automata_config["default_alphabet_size"]

    # PAA kaç segmente indirecek?
    # Başlangıç için sabit ve pratik bir değer kullanıyoruz.
    # Parametre analizi kısmında bunu ayrıca genişletebiliriz.
    n_segments = 300

    # Automata path probability için kaç pattern art arda değerlendirilecek?
    sequence_size = 3

    train_pc1, y_train, val_pc1, y_val, test_pc1, y_test = prepare_batadal_automata_data(config)

    train_patterns, train_paa, train_symbols = convert_series_to_patterns(
        series=train_pc1,
        n_segments=n_segments,
        alphabet_size=alphabet_size,
        window_size=window_size
    )

    val_patterns, val_paa, val_symbols = convert_series_to_patterns(
        series=val_pc1,
        n_segments=n_segments,
        alphabet_size=alphabet_size,
        window_size=window_size
    )

    test_patterns, test_paa, test_symbols = convert_series_to_patterns(
        series=test_pc1,
        n_segments=n_segments,
        alphabet_size=alphabet_size,
        window_size=window_size
    )

    train_pattern_windows = create_pattern_windows(
        train_patterns,
        sequence_size=sequence_size
    )

    val_pattern_windows = create_pattern_windows(
        val_patterns,
        sequence_size=sequence_size
    )

    test_pattern_windows = create_pattern_windows(
        test_patterns,
        sequence_size=sequence_size
    )

    y_val_aligned = align_labels_to_pattern_windows(
        y=y_val,
        n_segments=n_segments,
        window_size=window_size,
        sequence_size=sequence_size
    )

    y_test_aligned = align_labels_to_pattern_windows(
        y=y_test,
        n_segments=n_segments,
        window_size=window_size,
        sequence_size=sequence_size
    )

    print("\nAutomata veri özeti:")
    print("Train PC1:", train_pc1.shape)
    print("Validation PC1:", val_pc1.shape)
    print("Test PC1:", test_pc1.shape)
    print("Train pattern sayısı:", len(train_patterns))
    print("Validation pattern sayısı:", len(val_patterns))
    print("Test pattern sayısı:", len(test_patterns))
    print("Train pattern window sayısı:", len(train_pattern_windows))
    print("Validation pattern window sayısı:", len(val_pattern_windows))
    print("Test pattern window sayısı:", len(test_pattern_windows))
    print("y_val_aligned:", y_val_aligned.shape)
    print("y_test_aligned:", y_test_aligned.shape)

    model = ProbabilisticAutomataModel(
        anomaly_threshold=1e-6,
        smoothing=0.0,
        default_probability=1e-12
    )

    model.fit(train_patterns)

    best_threshold, validation_metrics, validation_probabilities = find_best_automata_threshold(
        model=model,
        validation_pattern_windows=val_pattern_windows,
        y_val_aligned=y_val_aligned
    )

    model.anomaly_threshold = best_threshold

    print("\nValidation üzerinde en iyi automata threshold:")
    print(best_threshold)

    print("\nValidation automata metrikleri:")
    print(validation_metrics)

    explanations = model.predict(test_pattern_windows)

    y_pred = np.array([
        1 if explanation["decision"] == "anomaly" else 0
        for explanation in explanations
    ])

    test_metrics = calculate_classification_metrics(
        y_true=y_test_aligned,
        y_pred=y_pred
    )

    print("\nTest automata metrikleri:")
    print(test_metrics)

    print("\nİlk 3 explainability çıktısı:")
    for explanation in explanations[:3]:
        print(explanation)

    result = {
        "dataset": "BATADAL",
        "model": "ProbabilisticAutomata",
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "n_segments": n_segments,
        "sequence_size": sequence_size,
        "selected_threshold": best_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "state_count": len(model.states),
        "transition_count": sum(
            len(next_states)
            for next_states in model.transition_probabilities.values()
        ),
        "sample_explanations": explanations[:5]
    }

    save_json_result(
        result=result,
        output_dir=outputs_config["metrics_dir"],
        file_name="batadal_automata_result.json"
    )

    return result


if __name__ == "__main__":
    run_batadal_automata_experiment()