import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 Model (pretrained)
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Open webcam
cap = cv2.VideoCapture(1)

# Initialize tracking variables
tracker = None
tracking = False
target_class = "person"  # Change this to the object you want to track

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
                class_id = int(box.cls[0])
                label = model.names[class_id]  # Get object label
                conf = float(box.conf[0])  # Confidence score

                if label == target_class and conf > 0.6:  # Track only the selected object
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
                    w, h = x2 - x1, y2 - y1  # Convert to (x, y, width, height)

                    # Initialize CSRT Tracker
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (x1, y1, w, h))
                    tracking = True
                    print(f"Tracking initialized on: {label}")
                    break  # Only track the first detected object

    # If tracking, update tracker
    if tracking and tracker is not None:
        success, roi = tracker.update(frame)
        if success:
            x, y, w, h = map(int, roi)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            print("Tracking lost, switching back to YOLO...")
            tracking = False  # Re-detect if tracking fails

    # Show output
    cv2.imshow("Hybrid YOLO + CSRT Tracker", display_frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
