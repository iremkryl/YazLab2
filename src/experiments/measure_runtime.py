import time
import json
from pathlib import Path

import numpy as np

from src.config.config_loader import load_config
from src.utils.seed import set_global_seed

from src.experiments.run_lstm import (
    prepare_batadal_lstm_data,
    calculate_class_weights as calculate_lstm_class_weights
)
from src.models.lstm_model import (
    build_lstm_model,
    train_lstm_model,
    predict_lstm_model
)

from src.experiments.run_gru import (
    prepare_batadal_gru_data,
    calculate_class_weights as calculate_gru_class_weights
)
from src.models.gru_model import (
    build_gru_model,
    predict_gru_model
)

from src.experiments.run_automata import (
    prepare_batadal_automata_data,
    convert_series_to_patterns,
    create_pattern_windows
)
from src.models.automata_model import ProbabilisticAutomataModel


def round_seconds(value):
    """
    Süre değerlerini okunabilir hale getirir.
    """
    return round(float(value), 4)


def measure_lstm_runtime(config):
    """
    LSTM için tek seed üzerinden eğitim ve inference süresini ölçer.

    Not:
    Runtime karşılaştırmasının çok uzun sürmemesi için 5 seed yerine
    ilk seed değeri kullanılır.
    """
    seed = config["project"]["random_seeds"][0]
    set_global_seed(seed)

    deep_learning_config = config["deep_learning"]

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_batadal_lstm_data(config)

    input_shape = (X_train.shape[1], X_train.shape[2])
    class_weight = calculate_lstm_class_weights(y_train)

    model = build_lstm_model(
        input_shape=input_shape,
        lstm_units=64,
        dropout_rate=0.30,
        learning_rate=0.001
    )

    train_start = time.perf_counter()

    train_lstm_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=deep_learning_config["epochs"],
        batch_size=deep_learning_config["batch_size"],
        patience=deep_learning_config["early_stopping_patience"],
        class_weight=class_weight
    )

    train_end = time.perf_counter()

    inference_start = time.perf_counter()

    predict_lstm_model(
        model=model,
        X=X_test,
        threshold=0.5
    )

    inference_end = time.perf_counter()

    return {
        "model": "LSTM",
        "seed": seed,
        "training_time_seconds": round_seconds(train_end - train_start),
        "inference_time_seconds": round_seconds(inference_end - inference_start)
    }


def measure_gru_runtime(config):
    """
    GRU için tek seed üzerinden eğitim ve inference süresini ölçer.

    Not:
    Runtime karşılaştırmasının çok uzun sürmemesi için 5 seed yerine
    ilk seed değeri kullanılır.
    """
    seed = config["project"]["random_seeds"][0]
    set_global_seed(seed)

    deep_learning_config = config["deep_learning"]

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_batadal_gru_data(config)

    input_shape = (X_train.shape[1], X_train.shape[2])
    class_weight = calculate_gru_class_weights(y_train)

    model = build_gru_model(
        input_shape=input_shape,
        gru_units=64,
        dropout_rate=0.30,
        learning_rate=0.001
    )

    train_start = time.perf_counter()

    # run_gru.py dosyasındaki eğitim mantığına yakın olması için
    # doğrudan model.fit kullanıyoruz.
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=deep_learning_config["epochs"],
        batch_size=deep_learning_config["batch_size"],
        class_weight=class_weight,
        verbose=0
    )

    train_end = time.perf_counter()

    inference_start = time.perf_counter()

    predict_gru_model(
        model=model,
        X=X_test,
        threshold=0.5
    )

    inference_end = time.perf_counter()

    return {
        "model": "GRU",
        "seed": seed,
        "training_time_seconds": round_seconds(train_end - train_start),
        "inference_time_seconds": round_seconds(inference_end - inference_start)
    }


def measure_automata_runtime(config):
    """
    Olasılıksal otomata için eğitim ve inference süresini ölçer.

    Eğitim süresi:
    - transition probability tablosunun öğrenilmesi

    Inference süresi:
    - test pattern window'ları için karar üretimi
    """
    automata_config = config["automata"]

    window_size = automata_config["default_window_size"]
    alphabet_size = automata_config["default_alphabet_size"]
    n_segments = 300
    sequence_size = 3

    train_pc1, y_train, val_pc1, y_val, test_pc1, y_test = prepare_batadal_automata_data(config)

    train_patterns, _, _ = convert_series_to_patterns(
        series=train_pc1,
        n_segments=n_segments,
        alphabet_size=alphabet_size,
        window_size=window_size
    )

    test_patterns, _, _ = convert_series_to_patterns(
        series=test_pc1,
        n_segments=n_segments,
        alphabet_size=alphabet_size,
        window_size=window_size
    )

    test_pattern_windows = create_pattern_windows(
        test_patterns,
        sequence_size=sequence_size
    )

    model = ProbabilisticAutomataModel(
        anomaly_threshold=1e-6,
        smoothing=0.0,
        default_probability=1e-12
    )

    train_start = time.perf_counter()

    model.fit(train_patterns)

    train_end = time.perf_counter()

    inference_start = time.perf_counter()

    model.predict(test_pattern_windows)

    inference_end = time.perf_counter()

    return {
        "model": "Automata",
        "training_time_seconds": round_seconds(train_end - train_start),
        "inference_time_seconds": round_seconds(inference_end - inference_start)
    }


def save_runtime_summary(results):
    """
    Runtime ölçüm sonuçlarını JSON dosyasına kaydeder.
    """
    output_dir = Path("results/metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "runtime_summary.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "description": "Runtime values were measured on a single local run using the first random seed for LSTM and GRU.",
                "results": results
            },
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nRuntime sonucu kaydedildi: {output_path}")


def print_markdown_table(results):
    """
    README'ye doğrudan kopyalanabilecek Markdown tablo üretir.
    """
    print("\nREADME için Runtime Analizi tablosu:")
    print()
    print("| Model | Training Time (sn) | Inference Time (sn) | Açıklama |")
    print("|---|---:|---:|---|")

    descriptions = {
        "LSTM": "Tek seed üzerinden ölçülmüştür; class weight ve early stopping kullanılmıştır.",
        "GRU": "Tek seed üzerinden ölçülmüştür; class weight kullanılmıştır.",
        "Automata": "Transition probability tablosu ve test pattern karar süresi ölçülmüştür."
    }

    for result in results:
        model_name = result["model"]
        training_time = result["training_time_seconds"]
        inference_time = result["inference_time_seconds"]
        description = descriptions.get(model_name, "")

        print(
            f"| {model_name} | {training_time:.4f} | {inference_time:.4f} | {description} |"
        )


def main():
    config = load_config()

    results = []

    print("\nLSTM runtime ölçümü başlıyor...")
    results.append(measure_lstm_runtime(config))

    print("\nGRU runtime ölçümü başlıyor...")
    results.append(measure_gru_runtime(config))

    print("\nAutomata runtime ölçümü başlıyor...")
    results.append(measure_automata_runtime(config))

    save_runtime_summary(results)
    print_markdown_table(results)


if __name__ == "__main__":
    main()