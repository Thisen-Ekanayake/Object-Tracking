import cv2

# Global variables
drawing = False  
ix, iy, fx, fy = -1, -1, -1, -1  
roi_selected = False
tracker_initialized = False  

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, roi_selected, tracker_initialized

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y  

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_selected = True  
        tracker_initialized = False  

# Open webcam
cap = cv2.VideoCapture(0)

# Create the window
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)  
#cv2.setWindowProperty("Live Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.setMouseCallback("Live Feed", draw_rectangle)

tracker = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the camera feed
    frame = cv2.flip(frame, 1)  

    # Create a copy of the frame to show live drawing without affecting tracking
    display_frame = frame.copy()

    # Draw the ROI dynamically while selecting
    if drawing:
        cv2.rectangle(display_frame, (ix, iy), (fx, fy), (255, 0, 0), 2)

    # If ROI is selected and tracker is not initialized
    if roi_selected and not tracker_initialized:
        tracker = cv2.TrackerCSRT_create()
        roi = (ix, iy, fx - ix, fy - iy)  
        tracker.init(frame, roi)
        tracker_initialized = True
        roi_selected = False  

    # If tracker is initialized, update tracking
    if tracker_initialized:
        success, roi = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in roi]
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Tracking failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with selection updates
    cv2.imshow("Live Feed", display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
