import cv2

# Global variables
drawing = False  # True if the mouse is pressed
ix, iy, fx, fy = -1, -1, -1, -1  # Coordinates for ROI
roi_selected = False
tracker_initialized = False  # Track if the tracker is initialized

# Mouse callback function to draw the ROI
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, roi_selected, tracker_initialized

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y  # Reset fx, fy in case of new selection

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_selected = True  # ROI selection complete
        tracker_initialized = False  # Reset tracker for new selection

# Open webcam
cap = cv2.VideoCapture(0)

# Create the window
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)  # Allow resizing
#cv2.setWindowProperty("Live Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set fullscreen mode

cv2.namedWindow("Live Feed")
cv2.setMouseCallback("Live Feed", draw_rectangle)

tracker = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally

    # Draw the ROI dynamically while selecting
    if drawing:
        temp_frame = frame.copy()
        cv2.rectangle(temp_frame, (ix, iy), (fx, fy), (255, 0, 0), 2)
        cv2.imshow("Live Feed", temp_frame)
    else:
        cv2.imshow("Live Feed", frame)

    # If ROI is selected and tracker is not initialized
    if roi_selected and not tracker_initialized:
        # Create a new tracker every time a new ROI is selected
        tracker = cv2.TrackerCSRT_create()
        roi = (ix, iy, fx - ix, fy - iy)  # Convert to x, y, width, height
        tracker.init(frame, roi)
        tracker_initialized = True
        roi_selected = False  # Reset selection flag

    # If tracker is initialized, update the tracking
    if tracker_initialized:
        success, roi = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in roi]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the tracked frame
    cv2.imshow("Live Feed", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
