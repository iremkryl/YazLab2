from src.data.load_skab import load_skab, prepare_skab_features_and_target
from src.data.split_data import create_skab_group_kfold_splits

def main():
    # Buraya kendi SKAB klasör yolunu yazacaksın.
    # Bu klasörün içinde valve1 ve valve2 klasörleri olmalı.
    skab_path = "data/raw/SKAB"

    df = load_skab(skab_path)

    print("\nİlk 5 satır:")
    print(df.head())

    print("\nSütun adları:")
    for column in df.columns:
        print("-", column)

    print("\nsource_group değerleri:")
    print(df["source_group"].value_counts())

    print("\nsource_file örnekleri:")
    print(df["source_file"].value_counts().head())

    print("\nanomaly dağılımı:")
    print(df["anomaly"].value_counts(dropna=False))

    X, y = prepare_skab_features_and_target(df)

    print("\nModel girdisi X hazırlandı:")
    print(X.head())

    print("\nHedef y hazırlandı:")
    print(y.head())

    print("\nX boyutu:", X.shape)
    print("y boyutu:", y.shape)

    print("\nX sütunları:")
    for column in X.columns:
        print("-", column)
    
        print("\nSKAB GroupKFold split testi başlıyor...")

    splits = create_skab_group_kfold_splits(
        df=df,
        target_column="anomaly",
        group_column="source_file",
        n_splits=5
    )

    print("\nToplam fold sayısı:", len(splits))
    print("SKAB GroupKFold split testi tamamlandı.")

if __name__ == "__main__":
    main()