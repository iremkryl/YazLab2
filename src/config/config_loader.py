import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """
    config.yaml dosyasını okuyup Python dictionary olarak döndürür.

    Projede hard-coded değer kullanmamak için parametreleri config dosyasından okuyacağız.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config dosyası bulunamadı: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config