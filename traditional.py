import cv2
import numpy as np

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Split into upper & lower face
        lower_face = face[int(h/2):h, :]

        # Convert to HSV
        hsv = cv2.cvtColor(lower_face, cv2.COLOR_BGR2HSV)

        # Skin color range (approx)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        skin_pixels = cv2.countNonZero(skin_mask)
        total_pixels = lower_face.shape[0] * lower_face.shape[1]

        skin_ratio = skin_pixels / total_pixels

        # Edge detection
        edges = cv2.Canny(lower_face, 100, 200)
        edge_pixels = cv2.countNonZero(edges)
        edge_ratio = edge_pixels / total_pixels

        # Decision logic
        if skin_ratio > 0.35 and edge_ratio < 0.1:
            label = "No Mask"
            color = (0, 0, 255)
        else:
            label = "Mask"
            color = (0, 255, 0)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Label
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Highlight lower face
        cv2.rectangle(frame,
                      (x, y + int(h/2)),
                      (x + w, y + h),
                      (255, 0, 0), 1)

    # Instruction text
    cv2.putText(frame, "Press Q or ESC to Exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show window
    cv2.imshow("Traditional Mask Detection (Live)", frame)

    # Exit key handling (UPDATED)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27:
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
