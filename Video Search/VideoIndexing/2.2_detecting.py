import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt


#path to COCO model trainer
model_path= 'ssd/saved_model'

#Check f path exists
if os.path.exists(model_path):
    print(f'The file at {model_path} exists.')
    model=tf.saved_model.load(model_path)
else:
    print(f'The file at {model_path} does not exist.')

#COCO class labels
labels = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
          'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
          'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
          'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
          'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
          'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
          'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
          'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
          'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush','hair brush']


image_detections=[]
videoid= "wbWRWeVe1XE"

#Open at frame per second(fps=24)
for index in range(0, 8189, 24):

    #Open image and create image as an array for processing
    image= Image.open(f'./Frames/{index}.jpg')
    image_np= np.array(image)
    image_tensor= tf.convert_to_tensor(np.expand_dims(image_np,0), dtype=tf.uint8)

    #detection model for the image
    detection= model(image_tensor)
    
    #detecting the boxes classes an scores
    boxes = detection['detection_boxes'].numpy()
    classes = detection['detection_classes'].numpy().astype(int)
    scores = detection['detection_scores'].numpy()
    
    #for every object detected
    for i in range(classes.shape[1]):
        object={}
        object["videoid"]= videoid
        object["frame"]= index
        object["second"]= index/24
        
        #filter out low confidence to minimze incorrect data
        '''
        I did this because it was detecting objects having like 0.3 confidence
        for "something", like, for example, sink, but when I looked at the picture
        it was totally wrong.
        '''
        confidence = scores[0, i]
        if confidence<0.5:
            continue

        #Get the class id
        class_id = int(classes[0, i])
        object["ObjId"]= class_id

        # Get the class name from the labels list
        # If object does not exist in labels
        if class_id>= len(labels):
            continue
        class_name = labels[class_id]
        object["ObjClass"]=class_name
        
        #Confidence
        object["Confidence"]= confidence
    
        #image shape and coordinates inside frame
        h, w, _ = image_np.shape
        ymin, xmin, ymax, xmax = boxes[0, i]

        # Convert normalized coordinates to image coordinates
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)

        object["bboxinfo: xmin xmax ymin ymax"]= [xmin, xmax, ymin, ymax]

        image_detections.append(object)

#Write image detections to file
file=open("2.2_obj.txt", "w")
for obj in image_detections:
   for key in obj:
      file.write(f'{key}: {obj[key]}')
      file.write('\t')
   file.write('\n')
