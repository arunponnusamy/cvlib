import os
import cv2
from tensorflow.keras.utils import get_file

initialize = True
gd = None

class GenderDetection():

    def __init__(self):

        proto_url = 'https://download.cvlib.net/config/gender_detection/gender_deploy.prototxt'
        model_url = 'https://github.com/arunponnusamy/cvlib-files/releases/download/v0.1/gender_net.caffemodel'
        save_dir = os.path.expanduser('~') + os.path.sep + '.cvlib' + os.path.sep + 'pre-trained'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.proto = get_file('gender_deploy.prototxt', proto_url,
                                cache_subdir=save_dir) 
        self.model = get_file('gender_net.caffemodel', model_url,
                              cache_subdir=save_dir)

        self.labels = ['male', 'female']
        self.mean = (78.4263377603, 87.7689143744, 114.895847746)

        print('[INFO] Initializing gender detection model ..')
        self.net = cv2.dnn.readNetFromCaffe(self.proto, self.model)


    def detect_gender(self, face, enable_gpu):

        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), self.mean,
                                     swapRB=False)

        if enable_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
        self.net.setInput(blob)
        preds = self.net.forward()

        return (self.labels, preds[0])
    

def detect_gender(face, enable_gpu=False):

    global initialize, gd

    if initialize:
        gd = GenderDetection()
        initialize = False
        
    return gd.detect_gender(face, enable_gpu)
