import cv2
import mediapipe as mp
import pyautogui

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Main loop
while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Only handle the first hand

        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        # Check if thumb is above the index finger
        if thumb_tip.y < index_tip.y:
            pyautogui.click()

        # Move the mouse pointer with the index finger tip
        x = int(index_tip.x * frame.shape[1])
        y = int(index_tip.y * frame.shape[0])
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        screen_x = screen_w * index_tip.x
        screen_y = screen_h * index_tip.y
        pyautogui.moveTo(screen_x, screen_y)

    # Display the frame
    cv2.imshow('Hand Controlled Mouse', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cam.release()
cv2.destroyAllWindows()
