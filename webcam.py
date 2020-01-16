import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")
faceCascade = cv2.CascadeClassifier("FaceDetect/faces.xml")

faceid = 1

c = 0

id = 0

font = cv2.FONT_HERSHEY_SIMPLEX

name = ["Miguel", "Maria"]

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
        #c+=1
        cv2.rectangle(frame, (x, y), (x+w, y+h), (231, 0, 42), 2)
        cv2.rectangle(gray, (x, y), (x+w, y+h), (231, 0, 42), 2)
        cv2.imwrite("dataset/User." + str(faceid) + '.' + str(c) + ".jpg", gray[y:y+h,x:x+w])
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        if (conf < 100):
            id = name[id-1]
            conf = "  {0}%".format(round(100 - conf))
        else:
            id = "unknown"
            conf = "  {0}%".format(round(100 - conf))
        cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(conf), (x+5,y+h-5), font, 1, (255,255,0), 1)
            

    cv2.imshow('frame', frame)
    
    k = cv2.waitKey(30)
    if k == 27:
        break
    
    #if c >= 90:
        #break
    
    
cap.release()
cv2.destroyAllWindows()


