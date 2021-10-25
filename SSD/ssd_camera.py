import cv2
import numpy as np

# Set up the fix variables
SCALING_FACTOR = 127.5
SSD_INPUT_SIZE = 320
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3


def read_class(file_name="class_names") -> list:
    """
    This function reads all the classes in the COCO dataset
    Args:
        file_name: name of the file
    Returns:
        list: list of the class names
    """
    with open(file_name, "rt") as file:
        names = file.read().rstrip("\n").split("\n")
    return names


def show_detected_objects(image: list, boxes_to_keep: list, all_bounding_boxes: list,
                          object_names: list, class_ids: list):
    """
    Plots the bounding box, name of the object and model confidence
    Args:
        image (list): image array
        boxes_to_keep (list): array of box ids to keep
        all_bounding_boxes (list): array of all possible boxes
        object_names (list): list of objects in COCO dataset
        class_ids (list): model predicted class ID

    Returns:
        None
    """
    for index in boxes_to_keep:
        box = all_bounding_boxes[index[0]]

        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(image, object_names[class_ids[index[0]][0] - 1].upper(),
                    (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 255), 1)


# Read the class names in COCO dataset
class_names = read_class()

# create the neural network model
neural_network = cv2.dnn_DetectionModel("ssd_weights.pb", "ssd_mobilenet_coco_cfg.pbtxt")

# Running model on CPU
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Other parameters for the model
neural_network.setInputSize(SSD_INPUT_SIZE, SSD_INPUT_SIZE)
neural_network.setInputScale(1.0 / SCALING_FACTOR)
neural_network.setInputMean((SCALING_FACTOR, SCALING_FACTOR, SCALING_FACTOR))
neural_network.setInputSwapRB(True)

# Reading the video

capture = cv2.VideoCapture(0)

while True:
    # capturing the image and validating if we have captured the image
    image_validation, image = capture.read()

    # if no image is captured, we break the loop
    if not image_validation:
        break

    # Getting the model predicted class labels, confidence and bounding boxes
    class_label_ids, confidences, bboxs = neural_network.detect(image)

    # to pass the bounding boxes and confidence levels through the Non-Max-Suppression, we need to convert them to 1-D list
    bboxs = list(bboxs)
    confidences = np.array(confidences).reshape(1, -1).tolist()[0]
    box_to_keep = cv2.dnn.NMSBoxes(bboxs, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    # Plot the image recognition
    show_detected_objects(image, box_to_keep, bboxs, class_names, class_label_ids)

    cv2.imshow("SSD_VIDEO", image)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()