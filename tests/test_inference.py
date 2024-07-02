"""
Run this file to check if your python environment and
project repository are set up correctly.
"""

from ultralytics import YOLOv10
import supervision as sv
import cv2
import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project root directory

# Initialize model and predict on test image
model = YOLOv10(f"{ROOT_DIR}/model/weights/yolov10n.pt")
results = model(source=f"{ROOT_DIR}/test_img.png", conf=0.25)
print(results[0].boxes.xyxy)

# Use supervision library for visualization
detections = sv.Detections.from_ultralytics(results[0])
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Prepare annotated image
image = cv2.imread(f"{ROOT_DIR}/test_img.png", cv2.IMREAD_UNCHANGED)
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
