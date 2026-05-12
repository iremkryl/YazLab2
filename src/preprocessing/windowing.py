import numpy as np


def create_sequences(X, y, sequence_length: int = 20):
    """
    LSTM/GRU gibi zaman serisi modelleri için pencereleme yapar.

    Örnek:
    sequence_length = 20 ise model her tahminde önceki 20 zaman adımını görür.

    X sonucu:
    (örnek_sayısı, sequence_length, özellik_sayısı)

    y sonucu:
    Her pencerenin son zaman adımındaki hedef etiketi alınır.
    """
    X_values = np.asarray(X)
    y_values = np.asarray(y)

    X_sequences = []
    y_sequences = []

    for i in range(len(X_values) - sequence_length + 1):
        X_window = X_values[i:i + sequence_length]
        y_label = y_values[i + sequence_length - 1]

        X_sequences.append(X_window)
        y_sequences.append(y_label)

    return np.array(X_sequences), np.array(y_sequences)