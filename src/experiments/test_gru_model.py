from src.models.gru_model import build_gru_model


def main():
    input_shape = (20, 8)

    model = build_gru_model(
        input_shape=input_shape,
        gru_units=64,
        dropout_rate=0.30,
        learning_rate=0.001
    )

    print("GRU modeli başarıyla oluşturuldu.")
    model.summary()


if __name__ == "__main__":
    main()