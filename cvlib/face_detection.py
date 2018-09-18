# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
import cv2
import numpy as np
import os
from pkg_resources import resource_filename, Requirement


def detect_face(image, threshold=0.5):
    
    if image is None:
        return None

    # access resource files inside package
    prototxt = resource_filename(Requirement.parse('cvlib'),
                                               'cvlib' + os.path.sep + 'data' + os.path.sep + 'deploy.prototxt')
    caffemodel = resource_filename(Requirement.parse('cvlib'),
                                            'cvlib' + os.path.sep + 'data' + os.path.sep + 'res10_300x300_ssd_iter_140000.caffemodel')
    
    # read pre-trained wieights
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    (h, w) = image.shape[:2]

    # preprocessing input image
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    net.setInput(blob)

    # apply face detection
    detections = net.forward()

    faces = []
    confidences = []

    # loop through detected faces
    for i in range(0, detections.shape[2]):
        conf = detections[0,0,i,2]

        # ignore detections with low confidence
        if conf < threshold:
            continue

        # get corner points of face rectangle
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype('int')

        faces.append([startX, startY, endX, endY])
        confidences.append(conf)

    # return all detected faces and
    # corresponding confidences    
    return faces, confidences
