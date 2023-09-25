import os 
import cv2 
import numpy as np

people=[]
DIR = f"C:\\Users\\DELL\\Desktop\\Face recognition\\data\\DATA"
for name in os.listdir(DIR):
    people.append(name)

features= []
labels =[]
haar_cascade = cv2.CascadeClassifier('haar_frontal_face.xml')

def training():
     for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for image in os.listdir(path):
            image_path = os.path.join(path,image)

            image_array = cv2.imread(image_path)
            gray=cv2.cvtColor(image_array,cv2.COLOR_BGR2GRAY)
  
            face_rects = haar_cascade.detectMultiScale(gray,scaleFactor =1.5,minNeighbors = 2)
             
            for (x,y,w,h) in face_rects :
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
     
training()
print('-----------Training done-----------------')       
features = np.array(features,dtype='object')
labels   = np.array(labels)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)


