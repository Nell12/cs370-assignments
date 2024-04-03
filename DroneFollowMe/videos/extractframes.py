#Task2: extracting frames to process
import cv2
import os

#get frames of a video
video=["CyclistandvehicleTracking1",
       "Cyclistandvehicletracking2",
       "DroneTrackingVideo"]

num=1
for i in video:
    cap= cv2.VideoCapture(f'./videos/{i}.mp4')
    if not cap.isOpened():
        print('error')

    count=0
    #frames per second
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
            cv2.imwrite(f'./videos/videoFrame{num}/{count}.jpg', frame)
        
        count+=1
    num+=1
           