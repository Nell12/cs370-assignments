# Drone Follow Me
## Task 2: Object Detection
### Trained the model yolov8n.pt on the visDrone Dataset in the pretraining-model directory(simplified the images to only those that contain vehicles, bicycles and people to speed up the training process). 
### Results: Did not perform well on video 2 and 3 [can only detect car in video 2, and can only detect pedestrians in video 3]. It performs semigood on video 1, detecting the car most of the time. The resulting detections are stored in the videos directory, as well as the extractframes.py, used to extract franes from the videos.