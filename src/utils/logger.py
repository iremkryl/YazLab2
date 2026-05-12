import json
from pathlib import Path
from datetime import datetime


def save_json_result(result: dict, output_dir: str, file_name: str):
    """
    Deney sonucunu JSON formatında kaydeder.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_path = Path(output_dir) / file_name

    result["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4, ensure_ascii=False)

    print(f"Sonuç kaydedildi: {output_path}")