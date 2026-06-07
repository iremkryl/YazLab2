import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

from src.config.config_loader import load_config
from src.experiments.run_automata import (
    prepare_batadal_automata_data,
    convert_series_to_patterns,
    create_pattern_windows,
    align_labels_to_pattern_windows,
    find_best_automata_threshold
)
from src.models.automata_model import ProbabilisticAutomataModel
from src.utils.metrics import calculate_classification_metrics


METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(file_name):
    """
    results/metrics klasöründeki JSON sonuç dosyasını okur.
    """
    file_path = METRICS_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} bulunamadı. "
            "Önce ilgili deney dosyasını çalıştırmalısın."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_model_comparison():
    """
    LSTM, GRU ve Automata modellerinin Accuracy, Precision, Recall ve F1
    metriklerini tek grafikte karşılaştırır.
    """
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
    """
    Automata modeli için confusion matrix görselini üretir.
    """
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
    """
    LSTM modelinin farklı seed değerlerindeki F1-score değişimini çizer.
    """
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
    """
    GRU modelinin farklı seed değerlerindeki F1-score değişimini çizer.
    """
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


def build_automata_artifacts(
    window_size=None,
    alphabet_size=None,
    n_segments=300,
    sequence_size=3
):
    """
    Automata görselleri ve parametre analizi için BATADAL üzerinde
    otomata modelini yeniden kurar.

    Bu fonksiyon:
    - BATADAL verisini hazırlar
    - PCA ile PC1 serisini çıkarır
    - PAA/SAX/sliding window uygular
    - Automata modelini fit eder
    - Validation üzerinden threshold seçer
    - Test olasılıklarını ve metrikleri üretir
    """
    config = load_config()
    automata_config = config["automata"]

    if window_size is None:
        window_size = automata_config["default_window_size"]

    if alphabet_size is None:
        alphabet_size = automata_config["default_alphabet_size"]

    train_pc1, y_train, val_pc1, y_val, test_pc1, y_test = prepare_batadal_automata_data(config)

    train_patterns, _, _ = convert_series_to_patterns(
        series=train_pc1,
        n_segments=n_segments,
        alphabet_size=alphabet_size,
        window_size=window_size
    )

    val_patterns, _, _ = convert_series_to_patterns(
        series=val_pc1,
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

    test_probabilities = np.array([
        model.calculate_sequence_probability(pattern_window)
        for pattern_window in test_pattern_windows
    ])

    y_pred = (test_probabilities < best_threshold).astype(int)

    test_metrics = calculate_classification_metrics(
        y_true=y_test_aligned,
        y_pred=y_pred
    )

    artifacts = {
        "model": model,
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "n_segments": n_segments,
        "sequence_size": sequence_size,
        "best_threshold": best_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "y_test": y_test_aligned,
        "y_pred": y_pred,
        "test_probabilities": test_probabilities,
        "train_patterns": train_patterns,
        "test_pattern_windows": test_pattern_windows
    }

    return artifacts


def plot_precision_recall_curve():
    """
    Automata modeli için Precision-Recall eğrisi üretir.

    Not:
    Automata modelinde düşük path probability daha yüksek anomali ihtimali
    anlamına geldiği için anomaly_score = 1 - path_probability şeklinde kullanılır.
    """
    artifacts = build_automata_artifacts()

    y_true = artifacts["y_test"]
    path_probabilities = artifacts["test_probabilities"]

    anomaly_scores = 1.0 - path_probabilities

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        anomaly_scores
    )

    average_precision = average_precision_score(
        y_true,
        anomaly_scores
    )

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.title(f"Precision-Recall Curve - Automata (AP={average_precision:.3f})")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()


def plot_transition_probability_heatmap():
    """
    Automata transition probability heatmap görselini üretir.
    """
    artifacts = build_automata_artifacts()
    model = artifacts["model"]

    transition_probs = model.get_transition_probabilities()

    all_states = sorted(model.states)

    # Çok fazla state olursa grafik okunmaz olur.
    # Bu nedenle en çok çıkış geçişi olan ilk 25 state seçilir.
    selected_states = sorted(
        all_states,
        key=lambda state: len(transition_probs.get(state, {})),
        reverse=True
    )[:25]

    matrix = np.zeros((len(selected_states), len(selected_states)))

    for i, from_state in enumerate(selected_states):
        for j, to_state in enumerate(selected_states):
            matrix[i, j] = transition_probs.get(from_state, {}).get(to_state, 0.0)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Transition Probability")
    plt.xticks(range(len(selected_states)), selected_states, rotation=90, fontsize=8)
    plt.yticks(range(len(selected_states)), selected_states, fontsize=8)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.title("Automata Transition Probability Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "transition_probability_heatmap.png", dpi=300)
    plt.close()


def plot_automata_state_diagram():
    """
    Automata state diagram görselini üretir.

    Grafik çok kalabalık olmasın diye en yüksek olasılıklı ilk 30 geçiş çizilir.
    """
    artifacts = build_automata_artifacts()
    model = artifacts["model"]

    transition_probs = model.get_transition_probabilities()

    graph = nx.DiGraph()
    edges = []

    for from_state, next_states in transition_probs.items():
        for to_state, probability in next_states.items():
            edges.append((from_state, to_state, probability))

    # En yüksek olasılıklı ilk 30 geçiş seçilir.
    edges = sorted(edges, key=lambda item: item[2], reverse=True)[:30]

    for from_state, to_state, probability in edges:
        graph.add_edge(from_state, to_state, weight=probability)

    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1200
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=8
    )

    edge_widths = [
        1 + 3 * graph[u][v]["weight"]
        for u, v in graph.edges()
    ]

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=15,
        width=edge_widths
    )

    edge_labels = {
        (u, v): f"{d['weight']:.2f}"
        for u, v, d in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7
    )

    plt.title("Automata State Diagram")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "automata_state_diagram.png", dpi=300)
    plt.close()


def evaluate_automata_parameter(
    window_size=None,
    alphabet_size=None
):
    """
    Tek bir automata parametre ayarı için F1-score değerini hesaplar.
    """
    artifacts = build_automata_artifacts(
        window_size=window_size,
        alphabet_size=alphabet_size
    )

    return artifacts["test_metrics"]["f1"]


def plot_automata_parameter_sensitivity():
    """
    Window size ve alphabet size parametrelerinin F1-score üzerindeki etkisini çizer.

    Burada gerçek deney akışı çalıştırılır:
    - Window size 3, 4, 5, 6 denenir.
    - Alphabet size 3, 4, 5, 6 denenir.
    """
    config = load_config()
    automata_config = config["automata"]

    default_window_size = automata_config["default_window_size"]
    default_alphabet_size = automata_config["default_alphabet_size"]

    window_sizes = automata_config["window_sizes"]
    alphabet_sizes = automata_config["alphabet_sizes"]

    window_f1_scores = []
    alphabet_f1_scores = []

    print("\nWindow size parametre analizi başlıyor...")

    for window_size in window_sizes:
        f1 = evaluate_automata_parameter(
            window_size=window_size,
            alphabet_size=default_alphabet_size
        )

        window_f1_scores.append(f1)
        print(f"window_size={window_size}, f1={f1:.4f}")

    print("\nAlphabet size parametre analizi başlıyor...")

    for alphabet_size in alphabet_sizes:
        f1 = evaluate_automata_parameter(
            window_size=default_window_size,
            alphabet_size=alphabet_size
        )

        alphabet_f1_scores.append(f1)
        print(f"alphabet_size={alphabet_size}, f1={f1:.4f}")

    plt.figure(figsize=(9, 5))

    plt.plot(
        window_sizes,
        window_f1_scores,
        marker="o",
        label="Window Size"
    )

    plt.plot(
        alphabet_sizes,
        alphabet_f1_scores,
        marker="o",
        label="Alphabet Size"
    )

    plt.ylim(0, 1)
    plt.xlabel("Parameter Value")
    plt.ylabel("F1 Score")
    plt.title("Automata Parameter Sensitivity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "automata_parameter_sensitivity.png", dpi=300)
    plt.close()


def main():
    """
    Rapor için gerekli tüm görselleri üretir.
    """
    plot_model_comparison()
    plot_automata_confusion_matrix()
    plot_lstm_seed_f1()
    plot_gru_seed_f1()

    # Rapor isterlerinde özellikle beklenen ek görseller
    plot_precision_recall_curve()
    plot_automata_state_diagram()
    plot_transition_probability_heatmap()
    plot_automata_parameter_sensitivity()

    print("\nGrafikler oluşturuldu:")
    print(FIGURES_DIR / "batadal_model_comparison.png")
    print(FIGURES_DIR / "automata_confusion_matrix.png")
    print(FIGURES_DIR / "lstm_f1_by_seed.png")
    print(FIGURES_DIR / "gru_f1_by_seed.png")
    print(FIGURES_DIR / "precision_recall_curve.png")
    print(FIGURES_DIR / "automata_state_diagram.png")
    print(FIGURES_DIR / "transition_probability_heatmap.png")
    print(FIGURES_DIR / "automata_parameter_sensitivity.png")


if __name__ == "__main__":
    main()