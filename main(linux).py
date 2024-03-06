import cv2 as cv
import mediapipe as mp
import numpy as np
import subprocess

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Constants
gesture_threshold = 40  # Adjust as needed
volume_percentage = 50  # Initial volume percentage, adjust as needed

# Calculate distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return np.linalg.norm(
        np.array([landmark1.x, landmark1.y]) - np.array([landmark2.x, landmark2.y])
    )

# update volume based on hand positions
def update_volume(hand_landmarks_list):
    global volume_percentage

    if len(hand_landmarks_list) >= 2:
        thumb_tip_left = hand_landmarks_list[0].landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip_left = hand_landmarks_list[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        thumb_tip_right = hand_landmarks_list[1].landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip_right = hand_landmarks_list[1].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        distance_left = calculate_distance(thumb_tip_left, index_tip_left)
        distance_right = calculate_distance(thumb_tip_right, index_tip_right)

        # Modify the distances as needed
        distance_right *= 100
        distance_left *= 100

        average_distance = (distance_left + distance_right) / 2

        if average_distance < gesture_threshold:
            # Adjust the mapping to your desired range
            volume_percentage = int(np.interp(average_distance, [0, gesture_threshold], [0, 100]))

            # Update system volume using ALSA
            subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{volume_percentage}%'], stdout=subprocess.DEVNULL)

            print(f"Volume: {volume_percentage}%")

# Capture video from webcam (change index to change camera)
cap = cv.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_landmarks_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_landmarks_list.append(hand_landmarks)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

        update_volume(hand_landmarks_list)

    cv.putText(frame, f"Volume: {volume_percentage}%", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv.imshow("gesture volume modifier", frame)

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
