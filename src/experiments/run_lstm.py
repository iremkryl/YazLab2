from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd

from src.config.config_loader import load_config
from src.data.load_batadal import (
    load_batadal,
    split_batadal_time_ordered,
    prepare_batadal_features_and_target
)
from src.preprocessing.scaler import fit_transform_scaler, transform_with_scaler
from src.preprocessing.windowing import create_sequences
from src.models.lstm_model import (
    build_lstm_model,
    train_lstm_model,
    predict_lstm_model
)
from src.utils.metrics import calculate_classification_metrics, find_best_threshold_by_f1
from src.utils.logger import save_json_result
from src.utils.seed import set_global_seed


def prepare_batadal_lstm_data(config: dict):
    """
    BATADAL verisini LSTM modeline uygun hale getirir.

    Adımlar:
    1. Veri okunur.
    2. Zaman sıralı train/validation/test bölünür.
    3. DATETIME ve ATT_FLAG ayrılır.
    4. Normalizasyon sadece train üzerinde fit edilir.
    5. LSTM için sequence verisi oluşturulur.
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

    sequence_length = preprocessing_config["sequence_length"]

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled,
        y_train,
        sequence_length=sequence_length
    )

    X_val_seq, y_val_seq = create_sequences(
        X_val_scaled,
        y_val,
        sequence_length=sequence_length
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test_scaled,
        y_test,
        sequence_length=sequence_length
    )

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq


def calculate_class_weights(y_train):
    """
    Veri dengesiz olduğu için class weight hesaplar.

    BATADAL'da normal örnekler anomalilerden çok daha fazla olduğu için
    modelin sadece normal sınıfa kaymasını azaltmak amacıyla kullanılır.
    """
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight = {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }

    return class_weight


def run_single_lstm_experiment(seed: int):
    """
    Tek bir seed değeri için LSTM deneyini çalıştırır.
    """
    config = load_config()
    set_global_seed(seed)

    deep_learning_config = config["deep_learning"]
    outputs_config = config["outputs"]

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_batadal_lstm_data(config)

    input_shape = (X_train.shape[1], X_train.shape[2])

    print("\nLSTM veri şekilleri:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    print("input_shape:", input_shape)

    class_weight = calculate_class_weights(y_train)

    print("\nClass weight:")
    print(class_weight)

    model = build_lstm_model(
        input_shape=input_shape,
        lstm_units=64,
        dropout_rate=0.30,
        learning_rate=0.001
    )

    model.summary()

    history = train_lstm_model(
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

    # Validation olasılıkları alınır
    y_val_pred_default, y_val_prob = predict_lstm_model(
        model,
        X_val,
        threshold=0.5
    )

    print("\nValidation probability özeti:")
    print("Min:", float(np.min(y_val_prob)))
    print("Max:", float(np.max(y_val_prob)))
    print("Mean:", float(np.mean(y_val_prob)))

    # En iyi threshold validation F1-score'a göre seçilir
    best_threshold, validation_threshold_metrics = find_best_threshold_by_f1(
        y_true=y_val,
        y_prob=y_val_prob
    )

    print("\nValidation üzerinde en iyi threshold:")
    print(best_threshold)

    print("\nValidation threshold metrikleri:")
    print(validation_threshold_metrics)

    # Test verisi, validation üzerinden seçilen threshold ile değerlendirilir
    y_pred, y_prob = predict_lstm_model(
        model,
        X_test,
        threshold=best_threshold
    )

    print("\nTest probability özeti:")
    print("Min:", float(np.min(y_prob)))
    print("Max:", float(np.max(y_prob)))
    print("Mean:", float(np.mean(y_prob)))

    metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred
    )

    result = {
        "dataset": "BATADAL",
        "model": "LSTM",
        "seed": seed,
        "input_shape": list(input_shape),
        "epochs_ran": len(history.history["loss"]),
        "selected_threshold": best_threshold,
        "validation_threshold_metrics": validation_threshold_metrics,
        "test_metrics": metrics
    }

    file_name = f"batadal_lstm_seed_{seed}.json"

    save_json_result(
        result=result,
        output_dir=outputs_config["metrics_dir"],
        file_name=file_name
    )

    print("\nTest metrikleri:")
    print(metrics)

    return result

def run_all_lstm_experiments():
    """
    Config dosyasında belirtilen tüm random seed değerleri için
    LSTM deneylerini çalıştırır.

    Her seed için ayrı sonuç kaydedilir.
    En sonunda ortalama ve standart sapma hesaplanır.
    """
    config = load_config()
    seeds = config["project"]["random_seeds"]
    outputs_config = config["outputs"]

    all_results = []

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"LSTM deneyi başlıyor. Seed: {seed}")
        print("=" * 80)

        result = run_single_lstm_experiment(seed=seed)

        metrics = result["test_metrics"]

        row = {
            "seed": seed,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "selected_threshold": result["selected_threshold"],
            "epochs_ran": result["epochs_ran"]
        }

        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    summary = {
        "dataset": "BATADAL",
        "model": "LSTM",
        "seeds": seeds,
        "mean_metrics": {
            "accuracy": float(results_df["accuracy"].mean()),
            "precision": float(results_df["precision"].mean()),
            "recall": float(results_df["recall"].mean()),
            "f1": float(results_df["f1"].mean())
        },
        "std_metrics": {
            "accuracy": float(results_df["accuracy"].std()),
            "precision": float(results_df["precision"].std()),
            "recall": float(results_df["recall"].std()),
            "f1": float(results_df["f1"].std())
        },
        "all_seed_results": all_results
    }

    save_json_result(
        result=summary,
        output_dir=outputs_config["metrics_dir"],
        file_name="batadal_lstm_summary_all_seeds.json"
    )

    print("\nLSTM 5 seed özet sonucu:")
    print(results_df)

    print("\nOrtalama metrikler:")
    print(summary["mean_metrics"])

    print("\nStandart sapma metrikleri:")
    print(summary["std_metrics"])

    return summary


if __name__ == "__main__":
    run_all_lstm_experiments()