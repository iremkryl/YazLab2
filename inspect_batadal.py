from src.config.config_loader import load_config
from src.data.load_batadal import load_batadal


def main():
    config = load_config()
    batadal_path = config["datasets"]["batadal"]["raw_path"]

    df = load_batadal(batadal_path)

    print("\nİlk 5 satır:")
    print(df.head())

    print("\nSütun adları:")
    for column in df.columns:
        print("-", column)

    print("\nEksik değer sayıları:")
    print(df.isnull().sum())

    print("\nVeri tipi bilgisi:")
    print(df.info())

    print("\nSon sütunun değer dağılımı:")
    last_column = df.columns[-1]
    print("Son sütun:", last_column)
    print(df[last_column].value_counts(dropna=False))


if __name__ == "__main__":
    main()