from flask import Flask, render_template
import cv2
import argparse
import json
from ultralytics import YOLO
import supervision as sv
import numpy as np


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

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720],            
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def count(objects):
    count_dict = {}
    
    for _, _, class_id, _ in objects:
        class_name = model.model.names[class_id].replace("Kurang Masaks", "Kurang Masak")
        class_name = model.model.names[class_id].replace("Abnormals", "Abnormal")
        class_name = model.model.names[class_id].replace("Janjang Kosongs", "Janjang Kosong")
        class_name = model.model.names[class_id].replace("Masaks", "Masak")
        class_name = model.model.names[class_id].replace("Mentahs", "Mentah")
        class_name = model.model.names[class_id].replace("Terlalu Masaks", "Terlalu Masak")
        if class_name in count_dict:
            count_dict[class_name] += 1
        else:
            count_dict[class_name] = 1    


    return count_dict

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    # rtsp://admin:qwerty13@10.10.12.107:554/Streaming/channels/1
    # rtsp://admin:qwerty13@10.10.15.107:554/Streaming/channels/1
    cap = cv2.VideoCapture("tes1.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    ab = 0
    me = 0
    ma = 0
    v_str = ""
    i=0
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

     # Set the output size to 640x480
    output_width = 1440
    output_height = 900
 
    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        count_dict = count(detections)
        # Draw a rectangle and put the count text for each label
        
        cv2.putText(frame, 'Abnormal :', (2700, 75 + (1*40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)  
        cv2.putText(frame, 'JanKos   :', (2700, 75 + (2*40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)  
        cv2.putText(frame, 'KM       :', (2700, 75 + (3*40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)  
        cv2.putText(frame, 'Masak    :', (2700, 75 + (4*40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)  
        cv2.putText(frame, 'Mentah   :', (2700, 75 + (5*40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)  
        cv2.putText(frame, 'TM       :', (2700, 75 + (6*40)), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)  
        

        # area_1 = [(,),(,),(,),(,)]


        for k, v in count_dict.items():
            i = label_map.get(str(k), -1)
            if i == -1:
                continue
            cnt_str = f"{k}: {v}"
            v_str = cnt_str.split(": ")[1]


        # xcatat = f"{k}: {v} ,"
        
            if 'Abnormal' not in count_dict:
                ab = 0
            else:
                ab = count_dict['Abnormal']


            if 'Mentah' not in count_dict:
                me = 0
            else:
                me = count_dict['Mentah']

            if 'Masak' not in count_dict:
                ma = 0
            else:
                ma = count_dict['Masak']    

        with open('fruits.json', 'r') as f:
             a = json.load(f)


        # # # add a new dictionary to the list
        if ab == 0 and me == 0 and ma == 0: 
            new_dict = {'Abnormal': ab, 'Mentah': me, 'Masak': ma}
            a.append(new_dict)

        # # write the updated list to the JSON file
        with open('fruits.json', 'w') as f:
             json.dump(a, f)
            
        # cv2.rectangle(frame, (frame_width - 300,(100)), (frame_width - 50, (250)), [85, 45, 255], -1,  cv2.LINE_AA)
        # cv2.putText(frame, v_str, (2750, 77 + (i*40)), 0, 1, [255, 255, 255], thickness=4, lineType=cv2.LINE_AA)    

            
        

        # count_text = ", ".join([f"{k}: {v}" for k, v in count_dict.items()])
        # Write the counts to the file

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        # cv2.putText(frame, cnt_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame = cv2.resize(frame,(output_width, output_height))

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break
        
if __name__ == "__main__":
    main()