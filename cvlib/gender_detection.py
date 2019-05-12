import cv2
import os
import numpy as np
from keras.utils import get_file
from keras.models import load_model
from keras.preprocessing.image import img_to_array

is_initialized = False
model = None

def pre_process(face):

    face = cv2.resize(face, (96,96))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)

    return face

def detect_gender(face):

    global is_initialized
    global model

    labels = ['man', 'woman']

    
    if not is_initialized:

        print("[INFO] initializing ... ")
        
        dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"

        model_path = get_file("gender_detection.model", dwnld_link,
                              cache_dir= os.path.expanduser('~') + os.path.sep + '.cvlib' + os.path.sep + 'pre-trained') 

        model = load_model(model_path)

        is_initialized = True


    face = pre_process(face)

    conf = model.predict(face)[0]

    return (labels, conf)

        
