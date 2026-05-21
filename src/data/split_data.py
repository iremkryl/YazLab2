from sklearn.model_selection import GroupKFold


def create_skab_group_kfold_splits(
    df,
    target_column: str = "anomaly",
    group_column: str = "source_file",
    n_splits: int = 5
):
    """
    SKAB veri seti için GroupKFold split üretir.

    Amaç:
    - Aynı CSV dosyasına ait satırlar aynı fold içinde kalsın.
    - Aynı source_file hem train hem test tarafına düşmesin.

    Parametreler:
    - df: SKAB dataframe
    - target_column: hedef sütun, SKAB için anomaly
    - group_column: grup sütunu, SKAB için source_file
    - n_splits: kaç fold oluşturulacağı

    Dönen değer:
    - splits listesi
      Her eleman:
      {
          "fold": fold numarası,
          "train_indices": train satır indexleri,
          "test_indices": test satır indexleri
      }
    """

    if target_column not in df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target_column}")

    if group_column not in df.columns:
        raise ValueError(f"Grup sütunu bulunamadı: {group_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    groups = df[group_column]

    group_kfold = GroupKFold(n_splits=n_splits)

    splits = []

    for fold_number, (train_indices, test_indices) in enumerate(
        group_kfold.split(X, y, groups),
        start=1
    ):
        train_files = set(df.iloc[train_indices][group_column].unique())
        test_files = set(df.iloc[test_indices][group_column].unique())

        common_files = train_files.intersection(test_files)

        if common_files:
            raise ValueError(
                f"Data leakage var! Aynı dosyalar train ve test içinde bulundu: {common_files}"
            )

        split_info = {
            "fold": fold_number,
            "train_indices": train_indices,
            "test_indices": test_indices
        }

        splits.append(split_info)

        print(f"\nFold {fold_number}")
        print(f"Train satır sayısı: {len(train_indices)}")
        print(f"Test satır sayısı: {len(test_indices)}")
        print(f"Train dosya sayısı: {len(train_files)}")
        print(f"Test dosya sayısı: {len(test_files)}")
        print("Ortak dosya var mı?:", "YOK" if not common_files else common_files)

    return splits