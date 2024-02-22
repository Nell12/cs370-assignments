#Problem 2.1
import cv2
import os

cap= cv2.VideoCapture('../VideoLibrary/Videos/Why It\'s Usually Hotter In A City  Lets Talk  NPR.mp4')
if not cap.isOpened():
    print('error')

digit= len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

n=0

while True:
    ret, frame= cap.read()

    if ret:
        cv2.imwrite(f'{n}.jpg', frame)
        n+=1
    else:
        break

    
           