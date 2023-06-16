# MINI PROJECT
# HAND GESTURE RECOGNITION

import os
import cv2
from tensorflow.keras.models import load_model
import pyautogui


# failsafe for uncontrollable mouse
# pyautogui.FAILSAFE = False

# Loading model
model = load_model("model_1.h5")

# Printing model summary for visualization
model.summary()

# Output labels
lables = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thrteen", "fourteen", "fifteen", "sixteen", "seventeen", "eightteen", "nineteen"]

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    x1 = 100
    y1 = 100
    x2 = 350
    y2 = 350

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)

    roi = frame[y1:y2, x1:x2]

    roi = cv2.resize(roi, (64, 64))

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

    cv2.imshow("ROI", roi)

    # Preprocessed image to be fed to neural network for prediction
    img_tobe_predicted = roi.reshape(1, 64, 64, 1)

    result = model.predict(img_tobe_predicted)

    # print(result)
    # print(f"type: {type(result)}")

    prediction = lables[result.argmax()]

    # print(prediction)

    # Mouse movements
    if prediction == "nine": # Left movement
        pyautogui.move(-10, 0)
        print("LEFT MOVEMENT")
    elif prediction == "six" or prediction == "seven": # Right movement
        pyautogui.move(10, 0)
        print("RIGHT MOVEMENT")
    elif prediction == "fifteen": # Single click
        pyautogui.click()
        print("SINGLE CLICK")
    elif prediction == "four": # Up movemvent
        pyautogui.move(0, 10)
        print("DOWN MOVEMENT")
    elif prediction == "eight": # Down movement
        pyautogui.move(0, -10)
        print("UP MOVEMENT")
    elif prediction == "eleven": # Scroll up
        pyautogui.click(button='right')
        print("RIGHT MOVEMENT")

    if cv2.waitKey(10) and 0xFF == 27:
        break
   
    
cap.release()
cv2.destroyWindow()
