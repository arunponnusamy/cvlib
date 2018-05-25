# cvlib 
A high level easy-to-use open source Computer Vision library for Python.

## Installation
Provided the below python packages are installed, cvlib is completely pip installable.
* numpy
* opencv-python

`pip install cvlib`

To upgrade to the newest version
`pip install --upgrade cvlib`

## Face detection
Detecting faces in an image is as simple as just calling the function `detect_face()`. It will return the bounding box corners and corresponding confidence for all the faces detected.
### Example :

``` 
import cvlib as cv
faces, confidences = cv.detect_face(image) 
```
Seriously, that's all it takes to do face detection with `cvlib`. Underneath it is using OpenCV's `dnn` module with a pre-trained caffemodel to detect faces. 

Checkout `face_detection.py` in `examples` directory for the complete code. 
