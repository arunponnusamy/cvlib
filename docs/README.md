# cvlib 
A simple, high level, easy-to-use open source Computer Vision library for Python.

It was developed with a focus on enabling easy and fast experimentation. Being able to go from an idea to prototype with least amount of delay is key to doing good research.

Guiding principles of cvlib are heavily inspired from [Keras](https://keras.io) (deep learning library). 
* simplicity
* user friendliness
* modularity and 
* extensibility

## Installation
Provided the below python packages are installed, cvlib is completely pip installable.
* OpenCV
* Tensorflow

If you don't have them already installed, you can install through pip

`pip install opencv-python tensorflow`

`pip install cvlib`

To upgrade to the newest version
`pip install --upgrade cvlib`

Checkout the [github](https://github.com/arunponnusamy/cvlib) page for complete instructions.

## Face detection
Detecting faces in an image is as simple as just calling the function `detect_face()`. It will return the bounding box corners and corresponding confidence for all the faces detected.
### Example :

``` 
import cvlib as cv
faces, confidences = cv.detect_face(image) 
```
Seriously, that's all it takes to do face detection with `cvlib`. Underneath it is using OpenCV's `dnn` module with a pre-trained caffemodel to detect faces. 

Checkout the github repo to learn more. 

## Gender detection
Once face is detected, it can be passed on to `detect_gender()` function to recognize gender. It will return the labels (man, woman) and associated probabilities. 

### Example

`label, confidence = cv.detect_gender(face) `

Underneath `cvlib` is using an AlexNet-like model trained on [Adience dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender) by Gil Levi and Tal Hassner for their [CVPR 2015 ](https://talhassner.github.io/home/publication/2015_CVPR) paper.

Checkout `gender_detection.py` in `examples` directory for the complete code. 


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

Checkout the github repo to learn more about all the functionalities available in cvlib. 

### License
cvlib is released under MIT License.

### Help
For bugs and feature requests, feel free to file a GitHub issue. (Make sure to check whether the issue has been filed already)

For usage related how-to questions, please create a new question on [StackOverflow](https://stackoverflow.com/questions/tagged/cvlib) with the tag `cvlib`.
