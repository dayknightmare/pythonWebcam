import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

faceCascade = cv2.CascadeClassifier("FaceDetect/faces.xml")

c = 0

faceid = input("Id: ")

name = input("Name: ")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.8,
        minNeighbors=5,
        minSize=(20, 20),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        c+=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (231, 0, 42), 2)
        cv2.rectangle(gray, (x, y), (x+w, y+h), (231, 0, 42), 2)
        cv2.imwrite("dataset/" + name + "." + str(faceid) + '.' + str(c) + ".jpg", gray[y:y+h,x:x+w])
        
    cv2.imshow('frame', frame)
    
    k = cv2.waitKey(30)
    if k == 27:
        break
    
    if c >= 180:
        break
    
    
cap.release()
cv2.destroyAllWindows()


