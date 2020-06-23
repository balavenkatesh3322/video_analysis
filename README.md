# video_analysis
Have you used AWS rekognition service for video analysis? 
I have implemented a small video analysis as aws rekognition using Yolo V3 and OpenCV. It's working pretty well.
This will read data from video and detect object names. 

Install below required library in your local machine.

1) python 3.7
2) opencv 4.1.0
3) numpy 

## Download Pre-Trained Yolov3 Model file
Download the pre-trained YOLO v3 weights file from this [link](https://drive.google.com/file/d/1AECks3mc2Xwe2BjvNdC_QKiiKZF8wt35/view?usp=sharing) and place it in the current directory

## Quick Start
Generate video from webcam using webcam.py

 `$ python3 webcam.py`

 The above python file will read input from webcam and save in the videos folder.


We can Analyse video files and get detected object names using analyse.py

 `$ python3 analyse.py --video videos/video_0.7177935927284033.mp4`


This analyse python file using Yolov3 to detect objects from videos and save object names as JSON file in videos folder.


## Sample Output
I have uploaded sample json file results in videos folder.

{'remote', 'cup', 'cell phone', 'person'}
