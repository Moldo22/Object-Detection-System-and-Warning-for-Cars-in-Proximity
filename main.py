from ultralytics import YOLO
import cv2
import pyttsx3
import numpy as np
import random

model = YOLO("yolov8l.pt")
cap = cv2.VideoCapture("<path>")
engine = pyttsx3.init()
cadre_limit = 30

def is_car_near_vertical_edges(car_box, image_height):
   
    cx1, cy1, cx2, cy2 = car_box

    image_top_edge = 0  
    image_bottom_edge = image_height  

    car_top_edge = cy1
    car_bottom_edge = cy2

  
    dist_to_top = abs(image_top_edge - car_top_edge)
    dist_to_bottom = abs(image_bottom_edge - car_bottom_edge)

    ##TODO: Adjust the values based on your image resolution
    is_near_top = dist_to_top < 400
    is_near_bottom = dist_to_bottom < 200
    return is_near_top and is_near_bottom



total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_random_frames = 50
random_frames = sorted(random.sample(range(total_frames), num_random_frames))
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index not in random_frames:
        frame_index += 1
        continue

    results = model(frame)
    person_boxes = []
    car_boxes = []
    vocal_msgs = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == 'person' and conf < 0.6:
                person_boxes.append((x1, y1, x2, y2))

            elif label in ['car', 'vehicle'] and conf < 0.6:
                car_boxes.append((x1, y1, x2, y2))
                if (is_car_near_vertical_edges((x1, y1, x2, y2), frame.shape[0])):
                    print("pune text")
                    cv2.putText(frame, "Warning! Car", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if label in ['person', 'car', 'vehicle','traffic light']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            if label == 'traffic light':
                vocal_msgs.append(f"Traffic Light Detected")
               
    cv2.imshow("Object Detection", frame)
    if vocal_msgs:
        try:
            engine.stop()
            engine = pyttsx3.init()
            for msg in vocal_msgs:
                engine.say(msg)
            engine.runAndWait()
        except RuntimeError:
            pass

    frame_index += 1

    if cv2.waitKey(3000) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
