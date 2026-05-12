from src.config.config_loader import load_config
from src.data.load_batadal import load_batadal, split_batadal_time_ordered


def main():
    config = load_config()

    print("Config başarıyla okundu.")
    print("Proje adı:", config["project"]["name"])

    batadal_path = config["datasets"]["batadal"]["raw_path"]

    try:
        df = load_batadal(batadal_path)

        split_batadal_time_ordered(
            df,
            train_ratio=config["datasets"]["batadal"]["train_ratio"],
            validation_ratio=config["datasets"]["batadal"]["validation_ratio"],
            test_ratio=config["datasets"]["batadal"]["test_ratio"]
        )

    except FileNotFoundError as error:
        print("BATADAL dosyası henüz eklenmedi.")
        print(error)


if __name__ == "__main__":
    main()