# Import some module 
import cv2, os
import numpy as np
from PIL import Image

# Create a directory to save data of faces and trainning data result
wajahDir = "datawajah"
data_training = "data_training"

# Create a function to get Image Label from datawajah folder
def getImageLabel(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    faceIDs = []

    for imagePath in imagePaths:
        PILImg =  Image.open(imagePath).convert("L") # Convert to gray
        imgNum = np.array(PILImg, 'uint8')
        faceID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = faceDetector.detectMultiScale(imgNum)

        for (x, y, w, h) in faces:
            faceSamples.append(imgNum[y:y+h, x:x+w])
            faceIDs.append(faceID)

        return faceSamples, faceIDs

# create Face Recognize LBPH
faceRecognize = cv2.face.LBPHFaceRecognizer_create()
faceDetector = faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print("Trainning Data Wajah")


faces, IDs = getImageLabel(wajahDir)
faceRecognize.train(faces, np.array(IDs))
faceRecognize.write(data_training+'/trainning.xml')

print(' Sebanyak {0} data waja telah di trainning.'. format(len(np.unique(IDs))))