[![Downloads](http://pepy.tech/badge/cvlib)](http://pepy.tech/project/cvlib)  [![Gitter](https://badges.gitter.im/arunponnusamy/cvlib.svg)](https://gitter.im/arunponnusamy/cvlib?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# cvlib 
A high level easy-to-use open source Computer Vision library for Python.

## Installation
Provided the below python packages are installed, cvlib is completely pip installable.
* numpy 
* opencv-python 
* requests
* progressbar

`pip install numpy opencv-python requests progressbar`

`pip install cvlib`

To upgrade to the newest version
`pip install --upgrade cvlib`

**Note: Python 2.x is not supported** 

## Face detection
Detecting faces in an image is as simple as just calling the function `detect_face()`. It will return the bounding box corners and corresponding confidence for all the faces detected.
### Example :

``` 
import cvlib as cv
faces, confidences = cv.detect_face(image) 
```
Seriously, that's all it takes to do face detection with `cvlib`. Underneath it is using OpenCV's `dnn` module with a pre-trained caffemodel to detect faces. 

Checkout `face_detection.py` in `examples` directory for the complete code. 

### Sample output :

![](examples/images/face_detection_output.jpg)

## Object detection
Detecting common objects in the scene is enabled through a single function call `detect_common_objects()`. It will return the bounding box co-ordinates, corrensponding labels and confidence scores for the detected objects in the image.

### Example :

``` 
import cvlib as cv
from cvlib.object_detection import draw_bbox

bbox, label, conf = cv.detect_common_objects(img)

output_image = draw_bbox(img, bbox, label, conf)
```
Underneath it uses [YOLOv3](https://pjreddie.com/darknet/yolo/) model trained on [COCO dataset](http://cocodataset.org/) capable of detecting 80 [common objects](https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.txt) in context.

Checkout `object_detection.py` in `examples` directory for the complete code. 

### Sample output :

![](examples/images/object_detection_output.jpg)

## License
cvlib is released under MIT license.
