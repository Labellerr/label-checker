# ============================================================
# SmartQC — Warehouse-Specific YOLO Training (Improved)
# ============================================================


from ultralytics import YOLO
import torch
import os

# ============================================================
# CONFIG
# ============================================================

DATASET_YAML = "warehouse.yaml"

MODEL_SIZE = "yolov8m.pt"
# options:
# yolov8n.pt  -> fastest
# yolov8s.pt  -> balanced
# yolov8m.pt  -> better accuracy
# yolov8l.pt  -> high GPU usage

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 16

PROJECT_NAME = "smartqc_warehouse"
RUN_NAME = "warehouse_yolo_v2"

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLO model...")
model = YOLO(MODEL_SIZE)

# ============================================================
# TRAIN
# ============================================================

print("\nStarting warehouse-specific training...\n")

results = model.train(

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    data=DATASET_YAML,

    # --------------------------------------------------------
    # Core training
    # --------------------------------------------------------
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=8,

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    project=PROJECT_NAME,
    name=RUN_NAME,
    exist_ok=True,

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,

    # --------------------------------------------------------
    # Training stability
    # --------------------------------------------------------
    patience=20,
    cos_lr=True,
    warmup_epochs=3,
    amp=True,

    # --------------------------------------------------------
    # Augmentation
    # Helps reduce false positives
    # --------------------------------------------------------
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    degrees=5.0,
    translate=0.10,
    scale=0.40,
    shear=2.0,
    perspective=0.0005,

    flipud=0.0,
    fliplr=0.5,

    mosaic=1.0,
    mixup=0.10,
    copy_paste=0.20,

    # --------------------------------------------------------
    # Better small object handling
    # --------------------------------------------------------
    close_mosaic=10,

    # --------------------------------------------------------
    # FP Reduction Improvements
    # --------------------------------------------------------
    box=8.0,
    cls=2.0,
    dfl=1.5,

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    val=True,
    plots=True,
    save=True,
    save_period=10,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    seed=42,
    deterministic=True,

    # --------------------------------------------------------
    # Better convergence
    # --------------------------------------------------------
    pretrained=True,
    verbose=True
)

# ============================================================
# VALIDATION
# ============================================================

print("\nRunning validation...\n")

metrics = model.val(
    data=DATASET_YAML,
    split="val",
    conf=0.25,
    iou=0.5,
    plots=True
)

# ============================================================
# EXPORT BEST MODEL
# ============================================================

print("\nExporting best model...\n")

best_model_path = f"runs/detect/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"

trained_model = YOLO(best_model_path)

trained_model.export(format="onnx")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n===================================================")
print("WAREHOUSE YOLO TRAINING COMPLETE")
print("===================================================")

print(f"Best model saved at:")
print(best_model_path)

print("\nExpected Improvements:")
print("Better warehouse localization")
print("Reduced false positives")
print("Better rack/pallet distinction")
print("Improved IoU quality")
print("Better small-object detection")
print("Stronger object boundaries")
print("More reliable SmartQC validation")


print("===================================================\n")