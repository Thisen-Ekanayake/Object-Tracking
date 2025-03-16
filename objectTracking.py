import cv2

# Open webcam (use 0 for the default camera, or replace with a video file path)
cap = cv2.VideoCapture(0)

# Grab the first frame to allow ROI selection
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    exit()

# Select the ROI (Region of Interest) for tracking
roi = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Initialize tracker with the selected ROI
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Update the tracker and get the new bounding box position
    success, roi = tracker.update(frame)
    
    if success:
        # Draw the bounding box if the tracking is successful
        x, y, w, h = [int(v) for v in roi]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # If tracking fails, mark it with a red box
        cv2.putText(frame, "Tracking failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame with the tracking box
    cv2.imshow("Object Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()