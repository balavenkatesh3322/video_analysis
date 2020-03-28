import cv2
import datetime
from random import random

#file name with random generated number
file_name = "video_"+str(random())+".mp4"

#Capture video from webcam
vid_capture = cv2.VideoCapture(0)
vid_cod = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter("videos/"+file_name, vid_cod, 10.0, (640,480))

while(True):
     # Capture each frame of webcam video
     ret,frame = vid_capture.read()
     cv2.imshow("My cam video", frame)
     output.write(frame)
     if cv2.waitKey(1) &0XFF == ord('q'):
         break

vid_capture.release()
output.release()
cv2.destroyAllWindows()