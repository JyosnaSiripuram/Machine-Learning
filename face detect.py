import cv2

image = cv2.imread(r"C:\\Users\\DELL\\Desktop\\Face recognition\\data\\DATA\\Anushka Sharma\\Anushka Sharma_9.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

haar_cascade = cv2.CascadeClassifier('haar_front_face.xml')
face_rects = haar_cascade.detectMultiScale(gray,scaleFactor =1.2,minNeighbors = 1)

print('no.of faces detected',{len(face_rects)})
for (x,y,w,h) in face_rects:
    cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),thickness = 2)
    cv2.imshow('detected face',image)
    cv2.waitKey(0)

    