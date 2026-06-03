import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(file_name):
    with open(METRICS_DIR / file_name, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_model_comparison():
    lstm = load_json("batadal_lstm_summary_all_seeds.json")
    gru = load_json("batadal_gru_summary_all_seeds.json")
    automata = load_json("batadal_automata_result.json")

    models = ["LSTM", "GRU", "Automata"]

    metrics = {
        "Accuracy": [
            lstm["mean_metrics"]["accuracy"],
            gru["mean_metrics"]["accuracy"],
            automata["test_metrics"]["accuracy"],
        ],
        "Precision": [
            lstm["mean_metrics"]["precision"],
            gru["mean_metrics"]["precision"],
            automata["test_metrics"]["precision"],
        ],
        "Recall": [
            lstm["mean_metrics"]["recall"],
            gru["mean_metrics"]["recall"],
            automata["test_metrics"]["recall"],
        ],
        "F1": [
            lstm["mean_metrics"]["f1"],
            gru["mean_metrics"]["f1"],
            automata["test_metrics"]["f1"],
        ],
    }

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(10, 6))

    for i, (metric_name, values) in enumerate(metrics.items()):
        plt.bar(x + i * width, values, width, label=metric_name)

    plt.xticks(x + width * 1.5, models)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("BATADAL Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "batadal_model_comparison.png", dpi=300)
    plt.close()


def plot_automata_confusion_matrix():
    automata = load_json("batadal_automata_result.json")
    cm = np.array(automata["test_metrics"]["confusion_matrix"])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Automata Confusion Matrix - BATADAL")
    plt.colorbar()

    labels = ["Normal", "Anomaly"]
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "automata_confusion_matrix.png", dpi=300)
    plt.close()


def plot_lstm_seed_f1():
    lstm = load_json("batadal_lstm_summary_all_seeds.json")

    seeds = [str(row["seed"]) for row in lstm["all_seed_results"]]
    f1_scores = [row["f1"] for row in lstm["all_seed_results"]]

    plt.figure(figsize=(8, 5))
    plt.bar(seeds, f1_scores)
    plt.ylim(0, 1)
    plt.xlabel("Seed")
    plt.ylabel("F1 Score")
    plt.title("LSTM F1 Score Across Seeds")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lstm_f1_by_seed.png", dpi=300)
    plt.close()


def plot_gru_seed_f1():
    gru = load_json("batadal_gru_summary_all_seeds.json")

    seeds = [str(row["seed"]) for row in gru["all_seed_results"]]
    f1_scores = [row["f1"] for row in gru["all_seed_results"]]

    plt.figure(figsize=(8, 5))
    plt.bar(seeds, f1_scores)
    plt.ylim(0, 1)
    plt.xlabel("Seed")
    plt.ylabel("F1 Score")
    plt.title("GRU F1 Score Across Seeds")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gru_f1_by_seed.png", dpi=300)
    plt.close()


def main():
    plot_model_comparison()
    plot_automata_confusion_matrix()
    plot_lstm_seed_f1()
    plot_gru_seed_f1()

    print("Grafikler oluşturuldu:")
    print(FIGURES_DIR / "batadal_model_comparison.png")
    print(FIGURES_DIR / "automata_confusion_matrix.png")
    print(FIGURES_DIR / "lstm_f1_by_seed.png")
    print(FIGURES_DIR / "gru_f1_by_seed.png")


if __name__ == "__main__":
    main()