from tensorflow import keras
model = keras.models.load_model('modelemotion.h5')

import cv2
import numpy as np
from keras.preprocessing import image

model.load_weights('modelemotion.h5')


 
faceCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
    faces = faceCascade.detectMultiScale(imgGray,1.1,4)
 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        crop_img = imgGray[y:y+w,x:x+h]
        crop_img = cv2.resize(crop_img,(48,48))
        img_pixels = image.img_to_array(crop_img)
        img_pixels = np.expand_dims(img_pixels,axis = 0 )
        img_pixels = img_pixels/255
        
        prediction = model.predict(img_pixels)

        max_index = np.argmax(prediction[0])
        emotion = ('angry','disgust','fear','happy','sad','surprise','neutral')
        predicted_emotion = emotion[max_index]
        cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    
    cv2.imshow("Result", img)
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
cap.release() 

