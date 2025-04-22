# Object Detection System and Warning for Cars in Proximity

This project uses a YOLO (You Only Look Once) model to detect objects such as **cars**, **people**, **traffic lights**, and **traffic signs** in a video sequence. Additionally, the project warns the user when a **car is near them**.

## Project Goal

The primary goal of this project is to monitor and warn pedestrians about risky situations, such as:

- Being close to cars.
- Presence of traffic lights and pedestrian crossings that can be positioned at intersections.

This system is an example of using artificial intelligence to improve road safety.

## Feature Description

### 1. **Object Detection**  
   - **YOLOv8** is used to detect objects in a video, such as **cars**, **people** and **traffic lights**.
   - Each detected object is surrounded by a green rectangle, and the label and confidence score are displayed on the screen.

   **Testing with different variants of YOLOv8:**
   - **YOLOv8n** (the "nano" model, the fastest and lightest for devices with limited resources).
   - **YOLOv8m** (the "medium" model, a compromise between speed and accuracy).
   - **YOLOv8l** (the "large" model, the most accurate but the slowest among the tested variants).

   The models were tested to determine which one provides the best performance based on the application and available resources.

### 2. **Warning for Cars in Proximity**
   - If a car is near the upper or lower edge of the image, the application will display a warning message on the screen (e.g., "Warning! The car is near the upper edge").
   - The distances between the edges of the video frame and the car's edges are compared with a set threshold. If the distances are below the threshold, the car is considered to be near the edge.

### 3. **Voice Feedback**
   - Using the `pyttsx3` library, the application can provide real-time voice messages. These messages include information about traffic lights (e.g., "The traffic light is green") and warnings (e.g., "Warning! A person is near the car").

### For future versions **Traffic Sign Detection with GTSRB**
   - The **GTSRB** (German Traffic Sign Recognition Benchmark: [GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/data)) dataset is used for detecting traffic signs in images and videos.
   - The dataset contains labeled images of various types of traffic signs, which can be used to train a YOLO model for detecting them.

   **Commands for training YOLO with the GTSRB dataset:**
   - The GTSRB dataset can be used to train a YOLO model with the following command:
     ```bash
     yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
     ```
     This command trains the YOLOv8 Nano model on the GTSRB dataset for 50 epochs, using images of size 640x640.

    **Data Modeling Before Training:**
    - The `trafficModel.yaml` file is configured to allow the training of the YOLO model with the GTSRB dataset.

## Requirements

To run this project on your system, you need to have the following Python libraries installed:

- `numpy`: for efficient numerical data manipulation.
- `opencv-python`: for image processing and video frame management.
- `ultralytics`: for implementing YOLOv8.
- `pyttsx3`: for generating voice feedback.

### Installation

To install these libraries, run the following commands in your terminal:

```bash
pip install numpy opencv-python ultralytics pyttsx3
