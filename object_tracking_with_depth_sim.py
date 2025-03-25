import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolov5 import YOLOv5
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load YOLOv5 model for object detection
yolo_model = YOLOv5('yolov5s.pt')

# Initialize Deep SORT for object tracking
deep_sort = DeepSort()

# Constant depth value (you can change this to any value you prefer)
constant_depth_value = 10  # meters

# Function to track and identify the object
def track_object(frame, deep_sort, yolo_model):
    # Perform object detection using YOLOv5
    results = yolo_model(frame)

    # Extract the bounding boxes, confidence scores, and class labels
    boxes = results.xywh[0][:, :-1].cpu().numpy()  # Bounding boxes (x, y, w, h)
    confidences = results.xywh[0][:, -1].cpu().numpy()  # Confidence scores
    class_ids = results.names

    # Perform object tracking with Deep SORT
    trackers = deep_sort.update(boxes, confidences, class_ids)

    return trackers

# Function to display the tracking results with a fixed depth value
def display_tracking_results(frame, trackers):
    for tracker in trackers:
        x1, y1, x2, y2, track_id = tracker
        label = f"ID: {track_id} - Depth: {constant_depth_value}m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Track and identify the object
    trackers = track_object(frame, deep_sort, yolo_model)

    # Display the tracking results with a fixed depth value
    display_tracking_results(frame, trackers)

    # Display the frame
    cv2.imshow("Tracking", frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
