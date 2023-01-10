# Air-keyboard-using-CNN

## Problem Statement:

An air keyboard is a wireless keyboard which performs the same functionalities of a physical keyboard and takes the input from the users. The user gestures and draws out the individual letters of a text which is detected and the output is displayed on the screen. In our approach the system requires only a camera and a processor. The proposed approach works in 2 phases, where initially pose estimation is performed on the user’s hand, to find key landmarks of the hand. This information is used to create a virtual “scratchpad” in front of the user where the user swipe types the letter in the air to finally create a swipe letter frame. The swiped letter frame then passes through a trained alphabet recognition model to detect the letter in its ASCII facilitating in typing the intended letter. 

## Code and Explanation:
Generate_dataset.py - We’ve used a POSNET model to detect the pose of the hand. Using this, we find the index-finger tip coordinates and keep tracking it. The points where the coordinates lay are pushed inside a list and between each consecutive point a line is drawn. When viewed in real-time, it looks like drawing letters on a black screen. We then store this black screen image with a letter to its respective letter folder.

Train.py - We’ve experimented with 2 different CNN architectures to find its use in handwritten letter detection. The first one was a simple CNN architecture with no additional layers to improve the accuracy. This model had an accuracy of 99%, but was not robust enough when tried in real-time. The second architecture is a modified version of LeNet5 architecture used to detect MNIST letters, which we modified to detect letters. This had similar accuracy scores but was much more resistant to discrepancies while typing in real-time.

Test.py - We have used the same methodology used in Generate_dataset.py to detect the inder finger tip. This was used to draw the letter on a black test screen. This image is then sent to the model trained to finally display the letter drawn as the output.

## Model Architecture:


























