import os
import cv2 
import numpy as np

people=[]
DIR = f"C:\\Users\\DELL\\Desktop\\Face recognition\\data\\DATA"
for name in os.listdir(DIR):
    people.append(name)
    
haar_cascade = cv2.CascadeClassifier('haar_frontal_face.xml') #detects objects in the image
features=np.load('features.npy',allow_pickle=True)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
count = 0

cam=cv2.VideoCapture(0) 
i=0
while True:
   isTrue,frame = cam.read()
    
   folderpath = 'C:\\Users\\RamyaJyosna\\Desktop\\Face recognition\\data\\frames\\'
   framepath = 'frame' + str(i)+'.jpg'
   cv2.imwrite(os.path.join(folderpath,framepath),frame)
   img = frame
   gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detects faces in the image
   faces_rect = haar_cascade.detectMultiScale(gray,1.1,5)

   for (x,y,w,h) in faces_rect:
      faces_roi = gray[y:y+h,x:x+h]
      label,accuracy = face_recognizer.predict(faces_roi)
      print(f'Label = {people[label]} with a accuracy of {accuracy}')

      if features == faces_roi:
        cv2.putText(img,str(people[label]),(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),thickness=2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        
      else:
        cv2.putText(img,str(people[label])+str(i),(20,20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),thickness=2)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
        cv2.imwrite(os.path.join(folderpath,framepath),img)           
        
   i+=1
   cv2.imshow('Detected face',img)
    
   if  cv2.waitKey(20) & 0xFF == ord("q"):
    break
    
cam.release()
cv2.destroyAllWindows()

