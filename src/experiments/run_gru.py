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
from src.models.gru_model import build_gru_model
from src.utils.metrics import calculate_classification_metrics, find_best_threshold_by_f1
from src.utils.logger import save_json_result
from src.utils.seed import set_global_seed


def prepare_batadal_gru_data(config: dict):
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
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    return {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }


def run_single_gru_experiment(seed: int):
    config = load_config()
    set_global_seed(seed)

    deep_learning_config = config["deep_learning"]
    outputs_config = config["outputs"]

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_batadal_gru_data(config)

    input_shape = (X_train.shape[1], X_train.shape[2])

    print("\nGRU veri şekilleri:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)
    print("input_shape:", input_shape)

    class_weight = calculate_class_weights(y_train)

    model = build_gru_model(
        input_shape=input_shape,
        gru_units=64,
        dropout_rate=0.30,
        learning_rate=0.001
    )

    model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=deep_learning_config["epochs"],
        batch_size=deep_learning_config["batch_size"],
        class_weight=class_weight,
        verbose=1
    )

    y_val_prob = model.predict(X_val).ravel()

    best_threshold, validation_threshold_metrics = find_best_threshold_by_f1(
        y_true=y_val,
        y_prob=y_val_prob
    )

    y_test_prob = model.predict(X_test).ravel()
    y_pred = (y_test_prob >= best_threshold).astype(int)

    metrics = calculate_classification_metrics(
        y_true=y_test,
        y_pred=y_pred
    )

    result = {
        "dataset": "BATADAL",
        "model": "GRU",
        "seed": seed,
        "input_shape": list(input_shape),
        "epochs_ran": len(history.history["loss"]),
        "selected_threshold": best_threshold,
        "validation_threshold_metrics": validation_threshold_metrics,
        "test_metrics": metrics
    }

    save_json_result(
        result=result,
        output_dir=outputs_config["metrics_dir"],
        file_name=f"batadal_gru_seed_{seed}.json"
    )

    print("\nGRU test metrikleri:")
    print(metrics)

    return result


def run_all_gru_experiments():
    config = load_config()
    seeds = config["project"]["random_seeds"]
    outputs_config = config["outputs"]

    all_results = []

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"GRU deneyi başlıyor. Seed: {seed}")
        print("=" * 80)

        result = run_single_gru_experiment(seed=seed)
        metrics = result["test_metrics"]

        all_results.append({
            "seed": seed,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "selected_threshold": result["selected_threshold"],
            "epochs_ran": result["epochs_ran"]
        })

    results_df = pd.DataFrame(all_results)

    summary = {
        "dataset": "BATADAL",
        "model": "GRU",
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
        file_name="batadal_gru_summary_all_seeds.json"
    )

    print("\nGRU 5 seed özet sonucu:")
    print(results_df)

    return summary


if __name__ == "__main__":
    run_all_gru_experiments()