from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC


def build_lstm_model(
    input_shape,
    lstm_units: int = 64,
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001
):
    """
    Binary anomaly detection için LSTM modeli oluşturur.

    input_shape:
    - (sequence_length, feature_count)

    Örnek:
    - sequence_length = 20
    - feature_count = 43
    - input_shape = (20, 43)
    """
    import tensorflow as tf

    model = Sequential([
        Input(shape=input_shape),

        LSTM(
            units=lstm_units,
            return_sequences=False
        ),

        Dropout(dropout_rate),

        Dense(
            units=32,
            activation="relu"
        ),

        Dropout(dropout_rate),

        Dense(
            units=1,
            activation="sigmoid"
        )
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc")
        ]
    )

    return model


def train_lstm_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 5,
    class_weight=None
):
    """
    LSTM modelini eğitir.

    Early stopping validation loss'a göre yapılır.
    """
    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1
    )

    return history


def predict_lstm_model(model, X, threshold: float = 0.5):
    """
    LSTM modelinden binary tahmin üretir.

    threshold:
    - 0.5 ve üzeri değerler anomali kabul edilir.
    """
    probabilities = model.predict(X)

    predictions = (probabilities >= threshold).astype(int).flatten()

    return predictions, probabilities.flatten()