This repository is about implementation of SSD algorithm using OpenCV in Python.

SSD stands for Single Shot multiDetector.
This model is based on the COCO dataset.

Model files:
ssd_mobilenet_coco_cfg.pbtxt - consists of SSD model configuration
ssd_weights.pb: SSD model weights


There are 3 .py files:
1. ssd_image - identifies objects in the image
2. ssd_video - identifies objects in the video
3. ssd_camera - identifies objects in the live camera on real-time basis

List of objects that the algorithm can detect are mentioned in the "class_name" file

Thank you !

You can contact me for critical feedback or for collaboration on shounak.python@gmail.com