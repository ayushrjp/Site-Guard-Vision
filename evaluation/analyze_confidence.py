
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "yolo_ppe_core.pt"
TEST_IMAGES = ROOT / "dataset" / "test" / "images"

model = YOLO(str(MODEL_PATH))

# Model class IDs corresponding to Roboflow classes
TARGET_CLASSES = {
    3: "helmet",
    8: "no-helmet",
    10: "no-vest",
    11: "person",
    13: "vest"
}

confidence_data = defaultdict(list)

image_paths = [
    p for p in TEST_IMAGES.iterdir()
    if p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }
]

print(f"Analyzing {len(image_paths)} images...")

for index, image_path in enumerate(image_paths, 1):

    result = model.predict(
        source=str(image_path),
        conf=0.001,
        imgsz=640,
        device="cpu",
        verbose=False
    )[0]

    for box in result.boxes:

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        if class_id in TARGET_CLASSES:
            confidence_data[class_id].append(confidence)

    print(f"[{index}/{len(image_paths)}] {image_path.name}")

print("\nCONFIDENCE DISTRIBUTION")
print("=" * 65)

thresholds = [0.10, 0.15, 0.25, 0.35, 0.50]

for class_id, class_name in TARGET_CLASSES.items():

    values = confidence_data[class_id]

    print(f"\n{class_name}")

    if not values:
        print("No predictions found.")
        continue

    print("Total predictions:", len(values))
    print("Mean confidence:", round(float(np.mean(values)), 4))
    print("Maximum confidence:", round(max(values), 4))

    for threshold in thresholds:

        count = sum(v >= threshold for v in values)

        print(
            f"Confidence >= {threshold:.2f}: {count}"
        )
