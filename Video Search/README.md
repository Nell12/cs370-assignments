# Aileen Ni
## Video Search Assignment
## Date of Submission: March 4 2024
## Instructions on how to run the code: 
### The first problem of the assignment is in the VideoLibrary directory 
#### The file video.py is run in python using pytube and youtube_transcript_ai.
### The second part of the assignment is in the VideoIndexing directory
#### File 2.1_decode.py breaks apart a video into frames by fps, and file 2.1_sample.py preprocess the frames and outputs them into a .npy file. The librarues used are cv2, os, and PIL libraries
#### File 2.2_detecting.py detects the objects in each frame, and uses the additional library tensorflow, and the ssd directory as the pre-trained model.