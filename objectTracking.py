import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 Model (pretrained)
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize tracking variables
tracker = None
tracking = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Mirror effect for better UX
    display_frame = frame.copy()

    # If not tracking, detect with YOLO
    if not tracking:
        results = model(frame)  # Run YOLO detection
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                label = model.names[int(box.cls[0])]  # Get object label
                conf = float(box.conf[0])  # Confidence score

                if conf > 0.6:  # Confidence threshold
                    tracker = cv2.TrackerCSRT_create()
                    roi = (x1, y1, x2 - x1, y2 - y1)  # Convert to ROI format
                    tracker.init(frame, roi)
                    tracking = True
                    break  # Only track the first detected object

    # If tracking, update tracker
    if tracking:
        success, roi = tracker.update(frame)
        if success:
            x, y, w, h = map(int, roi)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            tracking = False  # Re-detect if tracking fails

    # Show output
    cv2.imshow("Hybrid YOLO + CSRT Tracker", display_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()