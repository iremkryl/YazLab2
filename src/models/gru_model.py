from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC


def build_gru_model(
    input_shape,
    gru_units: int = 64,
    dropout_rate: float = 0.30,
    learning_rate: float = 0.001
):
    """
    Binary anomaly detection için GRU modeli oluşturur.

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

        GRU(
            units=gru_units,
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

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

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


def train_gru_model(
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
    GRU modelini eğitir.

    Önceki sürümde model 50 epoch boyunca devam edip overfit olabiliyordu.
    Bu sürümde EarlyStopping validation AUC'a göre yapılır.

    Böylece:
    - validation AUC en iyi olduğu noktadaki ağırlıklar saklanır,
    - model gereksiz yere 50 epoch boyunca overfit olmaz,
    - restore_best_weights=True sayesinde en iyi ağırlıklar geri yüklenir.
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


def predict_gru_model(model, X, threshold: float = 0.5):
    """
    GRU modelinden binary tahmin üretir.

    threshold:
    - threshold ve üzeri değerler anomali kabul edilir.
    """
    probabilities = model.predict(X).flatten()

    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities