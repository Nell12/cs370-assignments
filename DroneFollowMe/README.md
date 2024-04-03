# Drone Follow Me
## Python libraries used: os, cv2, pathlib, tqdm, PIL, ultralytics [for YOLOv8], filterpy, multiprocessing, pytube, re
## Task 2: Object Detection
### Trained the model yolov8n.pt on the visDrone Dataset in the PRETRAINING_MODEL directory  (simplified the images to only those that contain vehicles, bicycles and people to speed up the training process). 
### Results: stored in ./pretraining-model/analysis.txt
### The resulting detection frames are stored in the VIDEOS directory. extractframes.py, used to extract frames from the videos, and object_tracking.py to detect objects in each frame.