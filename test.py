import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

import string
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# reconstructed_model = tf.keras.models.load_model("typing_model")
reconstructed_model = tf.keras.models.load_model("typing_model_2")


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

prev = (0, 0)
points = []


def pre_process(img):
    img = cv2.resize(img, (50, 50))
    img = [img.astype('float32') / 255.0]
    img = tf.expand_dims(img, axis=-1)
    return img

alpha = string.ascii_lowercase
encode = {}
t = 1
for i in alpha:
    if i == "x":
        continue
    else:
        encode[t] = i
        t = t+1
encode[0] = '_'

print(encode)






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

    if key == ord("s"):
        cv2.imwrite("test/test.png", black_img)
        x = cv2.imread("test/test.png", 0)
        x = pre_process(x)
        n = reconstructed_model.predict(x).argmax(axis=-1)
        print(encode[n[0]])

        points.clear()
        black_img = np.zeros(imgRGB.shape, np.uint8)


    if key == ord("a"):
        points.clear()
        black_img = np.zeros(imgRGB.shape, np.uint8)


    if key == ord("q"):
        break

