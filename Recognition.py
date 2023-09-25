import numpy as np
import cv2 

people=['Akshay kumar','Alexandra Daddario','Amitabh Bachchan','Anushka Sharma']
#detects objects in the image
haar_cascade = cv2.CascadeClassifier('haar_frontal_face.xml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv2.imread(r"C:\Users\DELL\Desktop\Face recognition\data\DATA\Akshay kumar\Akshay Kumar_0.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detects faces in the image
faces_rect = haar_cascade.detectMultiScale(gray,1.1,5)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]

    label,accuracy = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a accuracy of {accuracy}')
    
    cv2.putText(img,str(people[label]),(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),thickness=2)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv2.imshow('Detected face',img)
cv2.waitKey(0)

