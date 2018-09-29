# cvlib 
A high level easy-to-use open source Computer Vision library for Python.

It was developed with a focus on enabling easy and fast experimentation. Being able to go from an idea to prototype with least amount of delay is key to doing good research.

Guiding principles of cvlib are heavily inspired from [Keras](https://keras.io) (deep learning library). 
* user friendliness
* modularity and 
* extensibility

## Installation
Provided the below python packages are installed, cvlib is completely pip installable.
* numpy 
* opencv-python 
* requests
* progressbar

`pip install numpy opencv-python requests progressbar`

`pip install cvlib`

## Face detection
Detecting faces in an image is as simple as just calling the function `detect_face()`. It will return the bounding box corners and corresponding confidence for all the faces detected.
### Example :

``` 
import cvlib as cv
faces, confidences = cv.detect_face(image) 
```
Seriously, that's all it takes to do face detection with `cvlib`. Underneath it is using OpenCV's `dnn` module with a pre-trained caffemodel to detect faces. 

Checkout `face_detection.py` in `examples` directory on github for the complete code. 


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

Checkout `object_detection.py` in `examples` directory on github for the complete code. 


## Issues
Feel free to create a new [issue](https://github.com/arunponnusamy/cvlib/issues) on [github](https://github.com/arunponnusamy/cvlib) if you are facing any difficulty.

## License
cvlib is released under MIT License.

## Contact
Feel free to drop an [email](http://arunponnusamy.com/contact) or reach out on [Twitter](twitter.com/ponnusamy_arun). 
