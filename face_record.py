# Import opencv module (cv2)
import cv2

# Create a directory to save data of faces
wajahDir = "datawajah"

# Camera Settings
cam = cv2.VideoCapture(0)
cam.set(3, 640) # Camera Width
cam.set(4, 480) # Camera Height

# Use cascade for Face and Eye Detector (https://github.com/opencv/opencv/tree/master/data/haarcascades)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector  = cv2.CascadeClassifier('haarcascade_eye.xml')

# Create input data from camera with nim
facesID = input('Masukan Nim Anda: ')
print('Lihat kamera dan Tunggu prosess recording')
ambilData = 1

# Loop and start the camera (will close if 'q' and 'esc' key will be pressed)
while True:
    # read the camera with read() function
    retV, frame = cam.read()

    # create rectangle to detect face
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)
    for(x, y, w, h) in faces:

        # get and save data faces
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        namaFile = 'wajah.'+str(facesID)+'.'+str(ambilData)+'.jpg'
        cv2.imwrite(wajahDir+'/'+namaFile, frame)
        ambilData += 1

        # create rectangle to detect eye
        grayColor = abuAbu[y:y+h, x:x+w]
        roiColor  = frame[y:y+h, x:x+w]
        eyes = eyeDetector.detectMultiScale(grayColor)
        for (xe, ye, we, he) in eyes:
            cv2.rectangle(roiColor, (xe, ye), (xe+we, ye+he),(0,0,255), 1)


    cv2.imshow("Webcam", frame)
    # cv2.imshow("Webcam1", abuAbu)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
    # will stop record when the data == 10
    elif ambilData>10:
        break

print("Recording Data Wajah Selesai")
cam.release()
cv2.destroyAllWindows()