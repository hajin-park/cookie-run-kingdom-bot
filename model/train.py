from ultralytics import YOLOv10
from roboflow import Roboflow
import os
from dotenv import load_dotenv

load_dotenv()

# Intialize Roboflow dataset
# rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
# project = rf.workspace(os.getenv("ROBOFLOW_WORKSPACE_ID")).project(
#     os.getenv("ROBOFLOW_MODEL_ID")
# )
# version = project.version(1)
# dataset = version.download("yolov8")  # v8 and v10 share formats

# Finetune pre-trained model
model = YOLOv10("weights/yolov10m.pt")

model.train(
    data=f"../datasets/Cookie-Run-Kingdom-1/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    device=0,
)
