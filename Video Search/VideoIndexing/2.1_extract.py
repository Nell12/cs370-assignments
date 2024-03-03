#Problem 2.1
import cv2
import os

#get frames of a video
video='WhatDoesHighQualityPreschoolLookLikeNPREd.mp4'
cap= cv2.VideoCapture(video)
if not cap.isOpened():
    print('error')

count=0
#frames per sond
fps= round(cap.get(cv2.CAP_PROP_FPS))
#print(fps)

while True:
    ret, frame= cap.read()

    if not ret:
       break
    
    #Minimize duplicate (similar frames)
    #Store frames every second
    
    if count%fps==0:
        #store video in directory as count.jpg
        cv2.imwrite(f'./Frames/{count}.jpg', frame)
    
    count+=1

           