from pathlib import Path
import pandas as pd


def load_skab(raw_path: str) -> pd.DataFrame:
    """
    SKAB veri setini okur.

    Bu projede yalnızca valve1 ve valve2 klasörleri kullanılacaktır.

    Yapılan işlemler:
    - valve1 ve valve2 klasörleri bulunur
    - içlerindeki tüm CSV dosyaları okunur
    - tüm dosyalar concat ile tek dataframe haline getirilir
    - source_group sütunu eklenir
        -> kaydın valve1 mi valve2 mi klasöründen geldiğini gösterir
    - source_file sütunu eklenir
        -> kaydın hangi CSV dosyasından geldiğini gösterir
    """

    path = Path(raw_path)

    if not path.exists():
        raise FileNotFoundError(f"SKAB klasörü bulunamadı: {raw_path}")

    groups_to_use = ["valve1", "valve2"]
    dataframes = []

    for group_name in groups_to_use:
        group_path = path / group_name

        if not group_path.exists():
            raise FileNotFoundError(
                f"{group_name} klasörü bulunamadı: {group_path}"
            )

        csv_files = list(group_path.glob("*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"{group_path} içinde CSV dosyası bulunamadı."
            )

        for csv_file in csv_files:
            df = pd.read_csv(csv_file, sep=";")

            # Sütun adlarında baş/son boşluk varsa temizle
            df.columns = df.columns.str.strip()

            # Bu satırların hangi klasörden geldiğini tutuyoruz
            df["source_group"] = group_name

            # Bu satırların hangi dosyadan geldiğini tutuyoruz
            df["source_file"] = csv_file.name

            dataframes.append(df)

    skab_df = pd.concat(dataframes, ignore_index=True)

    print("SKAB veri seti okundu.")
    print(f"Toplam veri boyutu: {skab_df.shape}")
    print("Kullanılan klasörler:", groups_to_use)
    print("Toplam dosya sayısı:", len(dataframes))

    return skab_df


def prepare_skab_features_and_target(
    df: pd.DataFrame,
    target_column: str = "anomaly"
):
    """
    SKAB veri setinden model girdisi X ve hedef değişken y üretir.

    SKAB için hedef sütun:
    - anomaly

    Model girdisine dahil edilmeyecek sütunlar:
    - datetime
    - changepoint
    - source_group
    - source_file
    - anomaly
    """

    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target_column}")

    y = df[target_column]

    columns_to_drop = [
        target_column,
        "datetime",
        "changepoint",
        "source_group",
        "source_file"
    ]

    # Veri setinde olmayan sütunları silmeye çalışırsak hata almamak için kontrol ediyoruz
    existing_columns_to_drop = [
        column for column in columns_to_drop
        if column in df.columns
    ]

    X = df.drop(columns=existing_columns_to_drop)

    return X, y