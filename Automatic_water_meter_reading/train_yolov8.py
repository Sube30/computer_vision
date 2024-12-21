#import required libraries
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO("yolov8m-seg.pt")

# Train the model with custom augmentations
model.train(
    data="data.yaml",  # Dataset YAML file
    epochs=50,                 # Number of training epochs
    batch=4,                  # Batch size
    imgsz=640,                 # Image size
    device=0,                  # Use GPU
    augment=True,              # Enable augmentations
    )
