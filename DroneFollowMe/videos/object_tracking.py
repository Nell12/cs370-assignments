from ultralytics import YOLO
import cv2

#load pretrained model
model=YOLO("./pretraining-model/best.pt")

#for each video and their corresponding frames
num=3
frames=[1320, 480, 2220]

for i in range(0,frames[num-1]+1,30):
    frame_path=f'./videos/videoFrame{num}/{i}.jpg'
    detections=model(frame_path)[0]
    
    #read frame
    frame=cv2.imread(frame_path)


    #[xmin,ymin,xmax,ymax,confidence_score, class_id]]

    # loop over the detections
    for data in detections.boxes.data.tolist():
        # confidence
        confidence = data[4]

        # where the object is detected in the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        #Draw the frame in green (0,255,0)
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), (0,255,0),2)

        #Get object class
        class_id= int(data[5])
        class_label= model.names[class_id]

        # Write class label and confidence
        text = f"{class_label}: {confidence:.2f}"
        cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #Write into videoDetect directory to see the results
    cv2.imwrite(f'./videos/videoDetect{num}/{i}.jpg', frame)