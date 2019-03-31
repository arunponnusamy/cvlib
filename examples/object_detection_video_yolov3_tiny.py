# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# object detection video example using tiny yolo model.
# usage: python object_detection_video_yolov3_tiny.py /path/to/video

# import necessary packages
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import sys

# open webcam
video = cv2.VideoCapture(sys.argv[1])

if not video.isOpened():
    print("Could not open video")
    exit()
    

# loop through frames
while video.isOpened():

    # read frame from webcam 
    status, frame = video.read()

    if not status:
        break

    # apply object detection
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')

    print(bbox, label, conf)

    # draw bounding box over detected objects
    out = draw_bbox(frame, bbox, label, conf, write_conf=True)

    # display output
    cv2.imshow("Real-time object detection", out)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
video.release()
cv2.destroyAllWindows()        
