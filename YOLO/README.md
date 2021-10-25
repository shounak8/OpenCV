This repository is about implementation of YOLO algorithm using OpenCV in Python.

YOLO stands for You Only Look Once.
This model is based on the COCO dataset.

Model files:
yolov4.weights - consists of model parameters
Link for yolov4.weights: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

yolov4.cfg - configuration of model and input required


There are 3 .py files:
1. yolo_image - identifies objects in the image
2. yolo_video - identifies objects in the video
3. yolo_camera - identifies objects in the live camera on real-time basis

List of objects that the algorithm can detect are:
    "PERSON",    "DOG",    "CUP",    "CHAIR",    "BACKPACK",    "HANDBAG",    "BALL",
    "TENNISRACKET",    "BOTTLE",    "TV",    "LAPTOP",    "MOUSE",    "REMOTE",    "KEYBOARD",
    "PHONE",    "CLOCK".

Thank you !

You can contact me for critical feedback or for collaboration on shounak.python@gmail.com