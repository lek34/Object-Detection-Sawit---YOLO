import cv2
import numpy as np
import argparse
from flask import Flask, render_template, Response

from ultralytics import YOLO
import supervision as sv

app = Flask(__name__)

count_data = []
model = YOLO("best.pt")

# Define a mapping from label strings to indices
label_map = {
    'Abnormal': 1,
    'Janjang Kosong': 2,
    'Kurang Masak': 3,
    'Masak': 4,
    'Mentah': 5,
    'Terlalu Masak': 6
}

def count(objects):
    count_dict = {}
    for _, _, class_id, _ in objects:
        class_name = model.model.names[class_id]
        if class_name in count_dict:
            count_dict[class_name] += 1
        else:
            count_dict[class_name] = 1
    return count_dict

def gen_frames():
    cap = cv2.VideoCapture("test1.mp4")
    
    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        count_dict = count(detections)
        # Draw a rectangle and put the count text for each label
        for k, v in count_dict.items():
            i = label_map.get(str(k), -1)
            if i == -1:
                continue
            cnt_str = f"{k}: {v}"
            cv2.rectangle(frame, (frame.shape[1] - 300, 45 + (i*40)), (frame.shape[1] - 50, 95 + (i*40)), [85, 45, 255], -1,  cv2.LINE_AA)
            cv2.putText(frame, cnt_str, (frame.shape[1] - 300, 75 + (i*40)), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)    
            count_data.append(cnt_str)
        
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        
        box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=2,
            text_scale=1
        )

        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        # Encode the frame as jpeg image
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)