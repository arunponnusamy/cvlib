# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# object detection webcam example
# usage: python object_detection_webcam.py

# right now YOLOv3 is being used for detecting objects.
# It's a heavy model to run on CPU. You might see the latency
# in output frames.
# To-Do: Add tiny YOLO for real time object detection

# import necessary packages
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    if not status:
        print("Could not read frame")
        exit()

    # apply object detection
    bbox, label, conf = cv.detect_common_objects(frame)

    print(bbox, label, conf)

    # draw bounding box over detected objects
    out = draw_bbox(frame, bbox, label, conf)

    # display output
    cv2.imshow("Real-time object detection", out)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()        
