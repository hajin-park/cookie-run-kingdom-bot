from screenshot import Screenshot
from ultralytics import YOLOv10
import supervision as sv

model = YOLOv10(f"../model/runs/detect/train8/weights/best.pt")


sct = Screenshot()


def main():
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        image = sct.get_screenshot()
        if not image.shape:
            print("Image not found")
            continue
        print(image.shape)
        results = model.predict(image)[0]
        detections = sv.Detections.from_ultralytics(results)

        boxed_image = bounding_box_annotator.annotate(
            scene=image, detections=detections
        )
        labeled_image = label_annotator.annotate(
            scene=boxed_image, detections=detections
        )

        sct.show_window([0, 560], labeled_image)


if __name__ == "__main__":
    main()
