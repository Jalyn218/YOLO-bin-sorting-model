'''
An algorithm to determine the material of an object

To determine the material used the YOLO uses the following:
1. Object detection
    Detects object and figures out which material list it belongs under

2. DIY spectrometer

3. Light sensor

NOTE: AI doesn't always complete all 3 levels - if its certainty is high enough it stops

Materials are split into the following:

Plastic
* Natural HDPE (Milks bottles)
* Natural PET (drinks bottles)
* Mixed plastic

Paper/Cardboard

Metal
* Tins
* Cans

Glass

Other/Trash

In the following code, only step 1 is executed as step 2 and 3 require input from a Raspberry Pi

'''


# Import libraries
import numpy as np
import cv2
from ultralytics import YOLO
import os

# YOLO functions and variables
model = YOLO("best.pt")

# Draw boxes on image
def draw_boxes(image):
    results = model.predict(image)  # Returns a list of objects
    confidence = results[0].boxes.conf # Confidence of model
    return results[0].plot()

# Capture real time feed from camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
else:
    print("Webcam accessed successfully!")

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video or error occurred.")
        break

    # Show the original webcam feed
    cv2.imshow("Video Frame", frame)

    # Create frame with boxes
    edited_frame = draw_boxes(frame)

    # Show edited webcam feed
    cv2.imshow("YOLO Frame", edited_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()
