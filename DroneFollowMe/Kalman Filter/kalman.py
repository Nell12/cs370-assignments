import numpy as np
import cv2
from matplotlib import pyplot as plt
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO

#initializing a Kalman Filter
def Kalman():
    kf= KalmanFilter(dim_x=4, dim_z=2)

    '''
    #x, F, H, P, R, Q

    x: ndarray(dim_x,1)
    F: ndarray (dim_x, dim_x) state transition matrix
    H: ndarray(dim_z, dim_x) measurement function
    P: ndarray (dim_x, dim_x) covariance matrix
    R: ndarray (dim_z, dim_z) measurement uncertainty
    Q: ndarray (dim_x, dim_x) process uncertainty
    '''

    #sampling time
    dt=0.1

    #Initial State
    kf.x= np.array([0,0,0,0])

    #state transition matrix
    kf.F= np.array([[1, 0, dt, 0],
                    [0, 1, 0, dt],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    #Define measurement mapping matrix
    kf.H= np.array([[1,0,0,0],
                    [0,1,0,0]])

    kf.Q= np.array([[dt**4/4, 0, dt**3/2, 0],
                    [0, dt**4/4, 0, dt**3/2],
                    [dt**3/2, 0, dt**2, 0],
                    [0, dt**3/2, 0, dt**2]])

    #Initial measurement noise covariance
    kf.R= np.array([[dt**2, 0],
                    [0, dt**2]])

    kf.P= np.eye(kf.F.shape[1])

    return kf

kf= Kalman()

#Setting up the videos for capture
videos= ["CyclistandvehicleTracking1", "CyclistandvehicleTracking2", "DroneTrackingVideo"]
video_path=f'../videos/{videos[1]}.mp4'
video= cv2.VideoCapture(video_path)

model = YOLO("../pretraining-model/models_gen2/model21.pt")


#mark positions of object located to update Kalman filter
#3D plot of positions against time
positions=[]
time=[]
frames=0

predictions=[]
timep=[]

#How many frames will the Kalman filter continue to predict until the object detection is lost
LIMIT=100
limit=0
predict=False

#Loop through each frame of video
while True:
    ret, frame= video.read()

    if not ret:
        break

    #run the YOLO model for object detection
    detections= model(frame)[0]

    #looping over detections and get only car object
    for data in detections.boxes.data.tolist():

        #Get class_id, class_label, and confidence
        class_id= int(data[5])
        class_label= model.names[class_id]

        if class_label.lower() != 'car':
            continue

        confidence=data[4]
        if confidence < 0.3:
            continue
        
        #Get bounding box
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        #Get middle position
        middle=((xmax+xmin)/2,(ymax+ymin)/2)

        #Update Kalman filter accordingly
        positions.append(middle)
        kf.update((middle[0], middle[1]))

        predict=True
        limit=0

        #Draw bounding box on object detection
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), (255,0,0), 2)
        text= f'{class_label}: {confidence:.2f}'
        cv2.putText(frame, text, (xmin, ymin-5),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 8)
        
        #cv2.circle(frame, (int(middle[0]), int(middle[1])), 10, (0,255,0), -1)
        
        #Append framenum
        time.append(frames)


    #update Kalman filter if predict is true
    if predict:
        if limit < LIMIT:
            kf.predict()
            predictions.append((kf.x[0], kf.x[1]))

            #draw prediction circle
            cv2.circle(frame, (int(kf.x[0]), int(kf.x[1])), 5, (255,0,255), -1)

            timep.append(frames)
            limit+=1
        #predict extra 100 frames if no object is detected for a while
        if limit >=100:
            predict=False
    

    frames+=1

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break


video.release()
cv2.destroyAllWindows()

#print actual vs predicted positions
print(positions)
print(predictions)


x,y= zip(*positions)
x1, y1= zip(*predictions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(x, y, time, color='red', marker='o', s=10, label="actual path")
ax.scatter(x1, y1, timep, color='blue', marker='_', s=5, label="predicted path")


# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot')

# Show plot
plt.legend()
plt.show()

