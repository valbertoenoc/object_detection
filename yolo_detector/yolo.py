import cv2
import os
import time
import numpy as np

class YoloDetector():
    def __init__(self):
        # yolo paths
        yolo_path = "yolo_detector\\yolo-coco"
        yolo_labels_path = os.path.sep.join([yolo_path, "coco.names"])
        yolo_weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
        yolo_config_path = os.path.sep.join([yolo_path, "yolov3.cfg"])
        
        # yolo params
        self.yolo_confidence_threshold = 0.5
        self.yolo_nms_threshold = 0.3

        # loading COCO labels
        self.LABELS = open(yolo_labels_path).read().strip().split('\n')

        # generate colors for each label
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype='uint8')

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

    def detect(self, image):
        (H, W) = image.shape[:2]

        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.yolo_confidence_threshold:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        return boxes, confidences, classIDs

    def draw_detection(self, image, boxes, confidences, classes):
        # apply non-maxima suppression to suppress weak, overlapping bounding
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.yolo_confidence_threshold,
            self.yolo_nms_threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in self.COLORS[classes[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classes[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        return image