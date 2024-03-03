# Aileen Ni
## Video Search Assignment
## Date of Submission: March 4 2024
## Instructions on how to run the code: 
### The first problem of the assignment is in the VideoLibrary directory 
#### The file video.py is run in python using pytube and youtube_transcript_ai.
### The second part of the assignment is in the VideoIndexing directory
#### File 2.2_extract.py breaks apart the video into frames by fps and stores them in the Frames directory (order is scrambled when uploaded to github for unknown reason), and file 2.1_sample.py preprocess the frames and outputs them into a .npy file. The librarues used are cv2, os, numpy, matplotlib and PIL libraries
#### File 2.2_detecting.py detects the objects in each frame, and uses the additional library tensorflow, and the ssd directory as the pre-trained model. It outputs the dections into 2.2_obj.txt
#### File 2.3_embedding.py trains the autoencoder using an additional library keras (on top of already mentioned libraries). The training result is outputted into vector_emb[1,2].png images.
