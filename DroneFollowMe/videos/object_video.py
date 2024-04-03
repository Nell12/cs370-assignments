from ultralytics import YOLO
import cv2

#tracks the object detection visually as it plays

# define some constants
#Confidence
CONFIDENCE_THRESHOLD = 0.3
RED = (255, 0, 0)

# initialize the video capture object
video_cap = cv2.VideoCapture("CyclistandvehicleTracking1.mp4")

#Write to the object detected
# initialize the video writer object
# load the pre-trained YOLOv8n model
model = YOLO("./models_gen2/last22.pt")


#Looping over video frames
while True:
    #compute the fps

    ret, frame= video_cap.read()

    #If there are no more frames
    if not ret:
        break

    #run the YOLO model
    detections= model(frame)[0]


    #[[xmin, ymin, xmax, ymax, confidence_score, class_id],...]

    #loop over detections
    for data in detections.boxes.data.tolist():
        confidence=data[4]

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # draw the bounding box on the frame
        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])

        #Get object class
        class_id= int(data[5])
        class_label= model.names[class_id]

        #draw rectangle
        cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), RED, 2)
        text= f'{class_label}: {confidence:.2f}'
        cv2.putText(frame, text, (xmin, ymin-5),
            cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 8)

    #show the frames
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
cv2.destroyAllWindows()