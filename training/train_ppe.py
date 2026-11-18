
from pathlib import Path
import shutil
import sys
import numpy as np


# ============================================================
# 1. NUMPY COMPATIBILITY FIX
# ============================================================

# Ultralytics 8.3.40 uses np.trapz during validation.
# Newer NumPy versions may not provide np.trapz.

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid


from ultralytics import YOLO


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

RUN_NAME = "siteguard_ppe_5class"

RUN_DIR = RUNS_DIR / RUN_NAME

WEIGHTS_DIR = RUN_DIR / "weights"

BEST_MODEL = WEIGHTS_DIR / "best.pt"

LAST_MODEL = WEIGHTS_DIR / "last.pt"

FINAL_MODEL = (
    PROJECT_ROOT / "models" / "siteguard_ppe_v2.pt"
)


# ============================================================
# 3. CPU TRAINING CONFIGURATION
# ============================================================

MODEL_NAME = "yolov8n.pt"

EPOCHS = 30

IMAGE_SIZE = 512

BATCH_SIZE = 2

PATIENCE = 10

DEVICE = "cpu"

WORKERS = 0

SEED = 42


# ============================================================
# 4. VERIFY DATASET
# ============================================================

def verify_dataset():

    print("\nDATASET VERIFICATION")
    print("=" * 60)

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    splits = ["train", "valid", "test"]

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    total_images = 0

    for split in splits:

        image_dir = (
            PROJECT_ROOT / "dataset" / split / "images"
        )

        label_dir = (
            PROJECT_ROOT / "dataset" / split / "labels"
        )

        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)

        if not label_dir.is_dir():
            raise FileNotFoundError(label_dir)

        images = [
            p for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in extensions
        ]

        labels = list(label_dir.glob("*.txt"))

        image_stems = {
            p.stem for p in images
        }

        label_stems = {
            p.stem for p in labels
        }

        missing_labels = image_stems - label_stems
        extra_labels = label_stems - image_stems

        print(f"\n{split.upper()} DATASET")
        print(f"Images: {len(images)}")
        print(f"Labels: {len(labels)}")
        print(f"Missing labels: {len(missing_labels)}")
        print(f"Extra labels: {len(extra_labels)}")

        if missing_labels or extra_labels:

            raise ValueError(
                f"Image/label filename mismatch in {split}"
            )

        total_images += len(images)

    print("\nTotal images:", total_images)
    print("Dataset verification successful.")


# ============================================================
# 5. TRAIN MODEL
# ============================================================

def train_model():

    print("\nLOADING YOLOv8n PRETRAINED MODEL")
    print("=" * 60)

    # Load pretrained YOLOv8n weights
    model = YOLO(MODEL_NAME)

    print("Model loaded successfully.")
    print("Starting five-class PPE training...")

    print("\nTRAINING CONFIGURATION")
    print("=" * 60)

    print("Architecture:", MODEL_NAME)
    print("Epochs:", EPOCHS)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH_SIZE)
    print("Device:", DEVICE)
    print("Workers:", WORKERS)
    print("Patience:", PATIENCE)

    # Start training
    results = model.train(

        # Dataset configuration
        data=str(DATA_YAML),

        # Training parameters
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,

        # CPU configuration
        device=DEVICE,
        workers=WORKERS,

        # Optimization
        optimizer="auto",
        pretrained=True,
        amp=False,

        # Reproducibility
        seed=SEED,
        deterministic=True,

        # Data augmentation
        mosaic=1.0,
        close_mosaic=10,
        fliplr=0.5,

        # Validation
        val=True,

        # Saving
        save=True,
        save_period=10,
        plots=True,

        # Memory management
        cache=False,

        # Output directory
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,

        # Logging
        verbose=True

    )

    print("\nTRAINING FINISHED.")

    return results


# ============================================================
# 6. SAVE BEST MODEL
# ============================================================

def save_best_model():

    print("\nSAVING BEST MODEL")
    print("=" * 60)

    if not BEST_MODEL.is_file():

        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST_MODEL}"
        )

    FINAL_MODEL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        BEST_MODEL,
        FINAL_MODEL
    )

    print("Best model copied successfully.")

    print("Source:", BEST_MODEL)

    print("Destination:", FINAL_MODEL)


# ============================================================
# 7. MAIN EXECUTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("SITEGUARDVISION")
    print("FIVE-CLASS PPE DETECTION TRAINING")
    print("=" * 60)

    print("Project:", PROJECT_ROOT)

    try:

        # Step 1: Verify dataset
        verify_dataset()

        # Step 2: Train model
        train_model()

        # Step 3: Save best checkpoint
        save_best_model()

        print("\n")
        print("=" * 60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except KeyboardInterrupt:

        print("\nTraining interrupted by user.")

        print(
            "Check the training weights folder "
            "for saved checkpoints."
        )

        raise

    except Exception as error:

        print("\nTRAINING FAILED")
        print("=" * 60)

        print("Error:", error)

        print(
            "\nCheck the error message and saved checkpoints "
            "before restarting training."
        )

        raise


# ============================================================
# 8. EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
