import cv2
import numpy as np

# Objects considered
classes = [
    "PERSON",
    "DOG",
    "CUP",
    "CHAIR",
    "BACKPACK",
    "HANDBAG",
    "BALL",
    "TENNISRACKET",
    "BOTTLE",
    "TV",
    "LAPTOP",
    "MOUSE",
    "REMOTE",
    "KEYBOARD",
    "PHONE",
    "CLOCK",
]

# Object id list is as per the COCO dataset
class_id_list = [
    0,
    16,
    41,
    56,
    24,
    26,
    32,
    38,
    39,
    62,
    63,
    64,
    65,
    66,
    68,
    74,
]

# Lets us define the constants
OBJ_PERSON = 0
OBJ_DOG = 16
OBJ_CUP = 41
OBJ_CHAIR = 56
OBJ_BACKPACK = 24
OBJ_HANDBAG = 26
OBJ_BALL = 32
OBJ_TENNISRACKET = 38
OBJ_BOTTLE = 39
OBJ_TV = 62
OBJ_LAPTOP = 63
OBJ_MOUSE = 64
OBJ_REMOTE = 65
OBJ_KEYBOARD = 66
OBJ_PHONE = 68
OBJ_CLOCK = 74

# algorithm will only detect objects if confidence level is above 30%
CONFIDENCE_THRESHOLD = 0.3

# algorithm will only detect objects if non max suppression (NMS) confidence level is above 30%
NMS_THRESHOLD = 0.3

# YOLO requires image object to be converted to a uniform dimension
# which is defined in the configuration file. In this case, it is 512*512
YOLO_FIG_DIM = 512


def find_objects(model_outputs: list) -> tuple:
    """
    This function takes forward propagation array of image as input and returns tuple of
    object boxes, object name & confidence level
    Args:
        model_outputs (list): array of image when passed through the yolo model
    Returns:
        tuple: tuple of object_type, bounding_box_location, class_id_values, confidence_values
    """
    bounding_box_location_values = []
    class_ids_values = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            if confidence > CONFIDENCE_THRESHOLD:  # 0.3 is the confidence threshold
                # prediction is in the form [x,y,w,h,conf,80 classes]
                # We increase the width height as per the YOLO config dimensions
                width, height = int(prediction[2] * YOLO_FIG_DIM), int(prediction[3] * YOLO_FIG_DIM)
                x, y = int(prediction[0] * YOLO_FIG_DIM - width / 2), int(
                    prediction[1] * YOLO_FIG_DIM - height / 2)

                bounding_box_location_values.append([x, y, width, height])
                class_ids_values.append(class_id)
                confidence_values.append(float(confidence))

    box_indices_to_keep = cv2.dnn.NMSBoxes(bounding_box_location_values, confidence_values,
                                           CONFIDENCE_THRESHOLD,
                                           NMS_THRESHOLD)  # considered nms_threshold = 0.3

    return box_indices_to_keep, bounding_box_location_values, class_ids_values, confidence_values
    # we have: box_indices_to_keep, bounding_box_location_values, class_ids_values, confidence_values


def show_detected_images(image: list, bounding_box_ids: list, all_bounding_boxes: list,
                         class_ids: list, confidence_values: list,
                         width_ratio: float, height_ratio: float):
    """
    This function plots bounding boxes on the objects along with name of the object and
    confidence level.
    Args:
        image (list): array of image
        bounding_box_ids(list): bounding box IDs
        all_bounding_boxes(list): bounding box indices for all possible bounding boxes
        class_ids (list): object IDs
        confidence_values (list): confidence level of the model about the object
        width_ratio (float): ratio of conversion of image width when passing through yolo
        height_ratio (float): ratio of conversion of image height when passing through yolo
    Returns:
        None
    """
    global classes, class_id_list
    for index in bounding_box_ids:
        bounding_box = all_bounding_boxes[index[0]]
        x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(
            bounding_box[3])
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)

        if class_ids[index[0]] == OBJ_PERSON:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_PERSON))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_DOG:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_DOG))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_CUP:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_CUP))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_CHAIR:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_CHAIR))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_BACKPACK:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_BACKPACK))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_HANDBAG:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_HANDBAG))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_BALL:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_BALL))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_TENNISRACKET:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_TENNISRACKET))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x - 100, y + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_BOTTLE:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_BOTTLE))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_TV:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_TV))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_LAPTOP:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_LAPTOP))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_MOUSE:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_MOUSE))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_REMOTE:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_REMOTE))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_KEYBOARD:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_KEYBOARD))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_PHONE:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_PHONE))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)
        elif class_ids[index[0]] == OBJ_CLOCK:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            class_list_index = int(class_id_list.index(OBJ_CLOCK))
            name_and_conf = classes[class_list_index] + str(
                int(confidence_values[index[0]] * 100)) + "%"
            cv2.putText(image, name_and_conf, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 255, 255), 1)


# Read the video
capture_video = cv2.VideoCapture("demo_video.mp4")

# use the parameters in yolov4.weights
neural_network = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")

# we will be using CPU as primary processing unit. For speedy processing, you can use GPU
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# We loop through all the frames the we are feeding the model
while True:
    # Read the video
    image_validation, image = capture_video.read()
    # Height and width of the image
    og_height, og_width = image.shape[0], image.shape[1]

    # If model doesn't capture the image, the loop breaks
    if not image_validation:
        break

    # convert image to BLOB
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (YOLO_FIG_DIM, YOLO_FIG_DIM), True, crop=False)
    neural_network.setInput(blob)

    # to see layer names
    layer_name = neural_network.getLayerNames()
    # get output layer indices (these start with index 1) using .getUnconnectedOutLayers():
    output_layers_indices = neural_network.getUnconnectedOutLayers()
    output_layers_names = [layer_name[index[0] - 1] for index in
                           output_layers_indices]  # received o/p as ['yolo_139', 'yolo_150', 'yolo_161']

    # pass the image via the model
    outputs = neural_network.forward(output_layers_names)

    # forward propagate array of image to get object boxes, object name & confidence level
    predicted_objects, bbox_locations, class_label_ids, conf_values = find_objects(outputs)

    # plot the boxes, object names and confidence levels on the image
    show_detected_images(image, predicted_objects, bbox_locations, class_label_ids, conf_values,
                         og_width / YOLO_FIG_DIM, og_height / YOLO_FIG_DIM)

    # return the frames with the object boxes
    cv2.imshow("sample", image)
    cv2.waitKey(1)

capture_video.release()
cv2.destroyAllWindows()
