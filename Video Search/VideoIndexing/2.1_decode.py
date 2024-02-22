#Problem 2.1
import cv2
import os

#get frames of a video
cap= cv2.VideoCapture('ShortwildlifevideoclipHD.mp4')
if not cap.isOpened():
    print('error')

digit= len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

n=0

while True:
    ret, frame= cap.read()

    if ret:
        #store video in directory as n.jpg
        cv2.imwrite(f'./VideoFrame/{n}.jpg', frame)
        n+=1
    else:
        break


           