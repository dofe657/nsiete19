import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('D:/opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
path = '../../translate/lehocki.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
        centerx = x + w / 2
        centery = y + h / 2
        nx = int (centerx - 150)
        ny = int (centery - 150)
        nr = int (150*2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        resized = cv2.resize(roi_color, (256,256))
        cv2.imwrite(path, resized)
        break