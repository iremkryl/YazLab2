from src.config.config_loader import load_config
from src.data.load_batadal import (
    load_batadal,
    split_batadal_time_ordered,
    prepare_batadal_features_and_target
)
from src.preprocessing.scaler import fit_transform_scaler, transform_with_scaler
from src.preprocessing.pca_transformer import fit_transform_pca, transform_with_pca
from src.preprocessing.windowing import create_sequences
from src.preprocessing.noise import add_gaussian_noise


def main():
    config = load_config()

    print("Config başarıyla okundu.")
    print("Proje adı:", config["project"]["name"])

    batadal_config = config["datasets"]["batadal"]
    preprocessing_config = config["preprocessing"]

    batadal_path = batadal_config["raw_path"]

    try:
        df = load_batadal(batadal_path)

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

        print("\nBATADAL model girdisi hazırlandı.")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        print("X_val:", X_val.shape)
        print("y_val:", y_val.shape)
        print("X_test:", X_test.shape)
        print("y_test:", y_test.shape)

        print("\nTrain hedef dağılımı:")
        print(y_train.value_counts())

        print("\nValidation hedef dağılımı:")
        print(y_val.value_counts())

        print("\nTest hedef dağılımı:")
        print(y_test.value_counts())

        # 1. Normalizasyon
        X_train_scaled, scaler = fit_transform_scaler(
            X_train,
            method=preprocessing_config["normalization"]
        )

        X_val_scaled = transform_with_scaler(X_val, scaler)
        X_test_scaled = transform_with_scaler(X_test, scaler)

        print("\nNormalizasyon tamamlandı.")
        print("X_train_scaled:", X_train_scaled.shape)
        print("X_val_scaled:", X_val_scaled.shape)
        print("X_test_scaled:", X_test_scaled.shape)

        # 2. PCA
        X_train_pca, pca = fit_transform_pca(
            X_train_scaled,
            n_components=preprocessing_config["pca_components"]
        )

        X_val_pca = transform_with_pca(X_val_scaled, pca)
        X_test_pca = transform_with_pca(X_test_scaled, pca)

        print("\nPCA tamamlandı.")
        print("X_train_pca:", X_train_pca.shape)
        print("X_val_pca:", X_val_pca.shape)
        print("X_test_pca:", X_test_pca.shape)

        # 3. LSTM/GRU için sequence oluşturma
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

        print("\nLSTM/GRU sequence verisi hazırlandı.")
        print("X_train_seq:", X_train_seq.shape)
        print("y_train_seq:", y_train_seq.shape)
        print("X_val_seq:", X_val_seq.shape)
        print("y_val_seq:", y_val_seq.shape)
        print("X_test_seq:", X_test_seq.shape)
        print("y_test_seq:", y_test_seq.shape)

        # 4. Gürültülü veri kontrolü
        X_train_noisy = add_gaussian_noise(
            X_train_scaled,
            seed=config["project"]["random_seeds"][0]
        )

        print("\nGaussian noise örneği oluşturuldu.")
        print("X_train_noisy:", X_train_noisy.shape)

    except FileNotFoundError as error:
        print("BATADAL dosyası henüz eklenmedi.")
        print(error)


if __name__ == "__main__":
    main()