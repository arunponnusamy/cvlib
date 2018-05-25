# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# face detection example
# usage: python face_detection.py <input_image>

# import necessary packages
import cvlib as cv
import sys
import cv2
import os 

# read input image
image = cv2.imread(sys.argv[1])

# apply face detection
faces, confidences = cv.detect_face(image)

print(faces)
print(confidences)

# loop through detected faces
for face,conf in zip(faces,confidences):

    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]

    # draw rectangle over face
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

# display output
# press any key to close window           
cv2.imshow("face_detection", image)
cv2.waitKey()

# save output
cv2.imwrite("face_detection.jpg", image)

# release resources
cv2.destroyAllWindows()

