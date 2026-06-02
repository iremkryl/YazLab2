from pathlib import Path
import pandas as pd


def find_batadal_file(raw_path: str) -> Path:
    """
    BATADAL klasörü içinde CSV dosyasını bulur.

    Bu projede BATADAL için yalnızca Training Dataset 2 kullanılacaktır.
    """
    path = Path(raw_path)

    if not path.exists():
        raise FileNotFoundError(f"BATADAL klasörü bulunamadı: {raw_path}")

    csv_files = list(path.glob("*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"{raw_path} içinde CSV dosyası bulunamadı. "
            "Training Dataset 2 dosyasını bu klasöre koymalısın."
        )

    if len(csv_files) > 1:
        print("Birden fazla CSV bulundu. İlk dosya kullanılacak:")
        for file in csv_files:
            print("-", file.name)

    return csv_files[0]


def load_batadal(raw_path: str) -> pd.DataFrame:
    """
    BATADAL Training Dataset 2 dosyasını okur.

    Not:
    BATADAL sütun adlarında başta boşluklar olabildiği için
    tüm sütun adları strip() ile temizlenir.
    """
    file_path = find_batadal_file(raw_path)
    df = pd.read_csv(file_path)

    # Sütun adlarındaki baş/son boşlukları temizle
    df.columns = df.columns.str.strip()

    print(f"BATADAL dosyası okundu: {file_path.name}")
    print(f"Veri boyutu: {df.shape}")

    return df


def split_batadal_time_ordered(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
    test_ratio: float = 0.20
):
    """
    BATADAL veri setini zaman sırası korunarak böler.

    Rastgele satır bazlı bölme yapılmaz.
    """
    total_ratio = train_ratio + validation_ratio + test_ratio

    if round(total_ratio, 2) != 1.00:
        raise ValueError("Train + validation + test oranları toplamı 1 olmalıdır.")

    n = len(df)

    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[train_end:validation_end].copy()
    test_df = df.iloc[validation_end:].copy()

    print("BATADAL zaman sıralı bölme tamamlandı.")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {validation_df.shape}")
    print(f"Test: {test_df.shape}")

    return train_df, validation_df, test_df


def prepare_batadal_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    time_column: str,
    normal_label: int = -999,
    anomaly_label: int = 1
):
    """
    BATADAL veri setinden model girdisi X ve hedef değişken y üretir.

    DATETIME model girdisine dahil edilmez.
    ATT_FLAG hedef değişken olarak alınır.

    ATT_FLAG:
    -999 -> 0, normal
    1    -> 1, anomali/saldırı
    """
    df = df.copy()

    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target_column}")

    if time_column not in df.columns:
        raise ValueError(f"Zaman sütunu bulunamadı: {time_column}")

    # Hedef etiketi binary hale getir
    y = df[target_column].replace({
        normal_label: 0,
        anomaly_label: 1
    })

    # Beklenmeyen etiket var mı kontrol et
    unexpected_labels = set(y.unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(f"Beklenmeyen hedef etiketleri bulundu: {unexpected_labels}")

    # Model girdisinden zaman ve hedef sütunlarını çıkar
    X = df.drop(columns=[time_column, target_column])

    return X,y