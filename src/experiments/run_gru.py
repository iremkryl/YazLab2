from src.data.load_skab import load_skab
from src.data.split_data import create_skab_group_kfold_splits
from src.automata.paa import apply_paa
from src.automata.sax import apply_sax
from src.models.gru_model import build_gru_model

import numpy as np


def main():

    # 1) Veri yükle
    df = load_skab("data/raw/SKAB")

    print("Dataset yüklendi")
    print(df.shape)

    # 2) Örnek veri seç
    series = df.select_dtypes(include=np.number).iloc[:200, 0].values

    # 3) PAA
    paa_output = apply_paa(series, n_segments=20)

    print("PAA tamam")

    # 4) SAX
    sax_output = apply_sax(
        paa_output,
        alphabet_size=5
    )

    print("SAX çıktısı:", sax_output)

    # 5) Dummy GRU
    model = build_gru_model(
        input_shape=(20, 1),
        gru_units=64,
        dropout_rate=0.3,
        learning_rate=0.001
    )

    print("GRU modeli hazır")

    model.summary()


if __name__ == "__main__":
    main()