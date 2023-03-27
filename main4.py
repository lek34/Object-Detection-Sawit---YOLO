import cv2
import json
from ultralytics import YOLO
import supervision as sv


model = YOLO("best.pt")


def main():
    # Load an image from file
    img = cv2.imread('sawitlewis.jpg')
    
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=1
    )

    result = model(img, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)

    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    annotated_img = box_annotator.annotate(
        scene=img, 
        detections=detections, 
        labels=labels
    )

    annotated_img = cv2.resize(annotated_img, (1440, 900))
    cv2.imshow("yolov8", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
