# Aileen Ni
# Drone Follow Me
## Python libraries used: os, cv2, pathlib, tqdm, PIL, ultralytics [for YOLOv8], filterpy, multiprocessing, pytube, re
## Task 2: Object Detection
### Trained the model yolov8n.pt on the visDrone Dataset in the PRETRAINING_MODEL directory  (simplified the images to only those that contain vehicles, bicycles and people to speed up the training process). 
### Results: model summary stored in ./pretraining-model/analysis.txt
### The resulting detection frames are stored in the VIDEOS directory. extractframes.py, used to extract frames from the videos, and object_tracking.py to detect objects in each frame.

## Task 3: Kalman Filter
### Results stored in ./Kalman Filter/Results
#### Since the thrid video's detection is not very good, only focused on the car in the first two videos. The Kalman tracker tracks the car's predicted position based on updates of the actual position by the object tracking model. However, there are many false positives, as seen in videoDetect1, that can throw off the Kalman prediction. 
#### To combat false positives, we can set the object positions in a way that if the next position differs significanlty, like the object was in some corner of a frame and moved all the way apart from their original position in the next frame, we would know that something has gone wrong with the object detector and ignore the detection. However, that would require a optimal object detector to be able to detect the original object, and the best way to combat false positive is to train the model on more datasets.
#### The Kalman filter can help with false positives by providing a framework for tracking objects overtime with state estimation, by measuring state of object based on previous observations and motion dynamics. Also, with the consistecy of the object's track, filtering out sporadic predictions, when false positives occur.