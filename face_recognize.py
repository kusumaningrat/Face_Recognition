# Import some module 
import cv2, os
from cv2 import data
import numpy as np

# Create a directory to save data of faces and trainning data result
wajahDir = "datawajah"
data_training = "data_training"

# Camera Settings
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Camera Width
cam.set(4, 480) # Camera Height

# Use cascade for Face and Eye Detector (https://github.com/opencv/opencv/tree/master/data/haarcascades)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognize = cv2.face.LBPHFaceRecognizer_create()

# Read data trainning.xml
faceRecognize.read(data_training+'/trainning.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak Diketahui', 'Kusuma Ningrat', 'Muhammad Zulhizmi', 'Egi Arya Guntara']

minWidth  = 0.1*cam.get(3)
minHeight = 0.1*cam.get(4)

# Loop and start the camera (will close if 'q' and 'esc' key will be pressed)
while True:
    # read the camera with read() function
    retV, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # create rectangle to detect face
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.2, 5, minSize=(round(minWidth), round(minHeight)),)

    for(x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        # Cek the image confidence from data_trainning and camera frame
        id, confidence = faceRecognize.predict(abuAbu[y:y+h, x:x+w])
        if confidence<=50:
            nameID = names[id]
            confidenceText = "{0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceText = "{0}%".format(round(100-confidence))

        # set for text to display when known and known person detect
        cv2.putText(frame, str(nameID), (x+5, y-5), font, 1, (255,255,255),2)
        cv2.putText(frame, str(confidenceText), (x+5, y+h-5), font, 1, (255,255,0),1)

    cv2.imshow("Recognize", frame)
    # cv2.imshow("Webcam1", abuAbu)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

print("Exit")
cam.release()
cv2.destroyAllWindows()