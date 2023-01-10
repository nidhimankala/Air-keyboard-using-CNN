import numpy as np
import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

prev = (0, 0)
points = []

name = 971

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    black_img = np.zeros(imgRGB.shape, np.uint8)
    fin = np.zeros(imgRGB.shape, np.uint8)

    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(landmark.landmark):
                h, w, c = black_img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:
                    points.append((cx, cy))

    for i in range(len(points)):
        if i > 0:
            cv2.line(black_img, points[i], points[i-1], (255,255, 255), 10)

    cv2.imshow("test", black_img)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if key == ord("s"):
        cv2.imwrite("data/_/"+str(name)+".png", black_img)
        print("data/_/"+str(name)+".png")
        points.clear()
        black_img = np.zeros(imgRGB.shape, np.uint8)

        name = name + 1

    if key == ord("a"):
        points.clear()
        black_img = np.zeros(imgRGB.shape, np.uint8)
