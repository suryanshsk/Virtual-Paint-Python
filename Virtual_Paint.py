import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Initialize the canvas
canvas = np.ones((480, 640, 3), dtype="uint8") * 255

# Brush parameters
brush_thickness = 3
brush_color = (114, 161, 223)  # SkyBlue Color color in BGR

# Drawing state
drawing = False
ix, iy = -1, -1

# Start video capture
cap = cv2.VideoCapture(0)

def select_color(key):
    global brush_color
    if key == ord('r'):
        brush_color = (0, 0, 255)  # Red
    elif key == ord('g'):
        brush_color = (0, 255, 0)  # Green
    elif key == ord('b'):
        brush_color = (255, 0, 0)  # Blue
    elif key == ord('y'):
        brush_color = (0, 255, 255)  # Yellow
    elif key == ord('k'):
        brush_color = (0, 0, 0)  # Black
    elif key == ord('w'):
        brush_color = (255, 255, 255)  # White (eraser)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 5:  # Index finger tip
                    if drawing:
                        cv2.line(canvas, (ix, iy), (cx, cy), brush_color, brush_thickness)
                    ix, iy = cx, cy

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Paint', canvas)
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF

    # Brush size control
    if key == ord('+') and brush_thickness < 20:
        brush_thickness += 1
    elif key == ord('-') and brush_thickness > 1:
        brush_thickness -= 1

    # Color selection
    select_color(key)

    # Clear canvas
    if key == ord('c'):
        canvas = np.ones((480, 640, 3), dtype="uint8") * 255

    # Start/Stop drawing
    if key == ord('d'):
        drawing = not drawing

    # Exit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
