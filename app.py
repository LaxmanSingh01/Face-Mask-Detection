import streamlit as st
import os
import numpy as np
import cv2
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

model = load_model(r'E:\Project for Portfolio\Face Mask Detection\vgg.h5')


def detect_mask(image):
    y_pred=model.predict_classes(image.reshape(1,224,224,3))
    return y_pred[0][0]


cap=cv2.VideoCapture(0)
st.title("Face Mask Detection")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
while run:
    ret,frame=cap.read()
    img=cv2.resize(frame,(224,224))
    y_pred=detect_mask(img)
    if y_pred == 0:
        cv2.putText(frame,'No Mask',(30,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
    if y_pred == 1:
        cv2.putText(frame,'Mask',(30,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    points=detector.detectMultiScale(gray)
    for (x, y, w, h) in points:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
    
 