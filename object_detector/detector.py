from PIL import Image
import numpy as np
import tensorflow as tf
import argparse
import os
import cv2
import sys
sys.path.append('object_detector/slim')

from objectdetection import draw_detected_objects
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_image_into_numpy_array(image):
  '''
    This function loads a single PIL image into memory converting it to numpy array
  '''
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

    
def load_graph(model_path):
    '''
        This function loads exported graph into memory for detection
    '''
    LABELS_PATH = os.path.join(model_path, 'labelmap.pbtxt')
    label_map   = label_map_util.load_labelmap(LABELS_PATH)
    num_classes = len(label_map.item._values)
    categories  = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    label       = categories[0]['name']
    GRAPH_PATH  = os.path.join(model_path, 'frozen_inference_graph.pb')

    # Load the graph
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.gfile.GFile(GRAPH_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)
    
    return detection_graph, sess, categories


def detect_object_path(detection_graph, sess, categories, image_path):
    '''
        This function uses loaded graph to perform detection on single image
    '''
    with detection_graph.as_default():
        # Preprocessing
        # image = Image.open(image_path).resize((300, 300)) # Resize image (prevent OOM)
        # image_np = load_image_into_numpy_array(image)
        image_np = cv2.imread(image_path)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})
        
        # Found?
        if scores[0][0] < 0.1:
            print('No object found.')

        # Plot
        # categories = [{'id': 1, 'name': 'RG-F'}]
        category_index = label_map_util.create_category_index(categories)
        max_score, max_box, max_class = draw_detected_objects(image_np, boxes, classes, scores, category_index, max_only=False)

    return image_np, max_score, max_box, max_class


def detect_object(detection_graph, sess, categories, image_np):
    '''
        This function uses loaded graph to perform detection on single image
    '''
    with detection_graph.as_default():
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})
        
        # # Found?
        # if scores[0][0] < 0.1:
        #     print('No object found.')

        # Plot
        # categories = [{'id': 1, 'name': 'RG-F'}]
        category_index = label_map_util.create_category_index(categories)
        max_score, max_box, max_class = draw_detected_objects(image_np, boxes, classes, scores, category_index, max_only=False)

    return image_np, max_score, max_box, max_class


def main():
    model_path = 'data\\trained_models\\inference_graph_RG-F'
    image_path = 'path\\to\\test_images\\RG-F_12-216-644-1267_.png'

    detection_graph, sess, categories = load_graph(model_path)
    results = detect_object_path(detection_graph, sess, categories, image_path)

    cv2.imshow('img', results[0])
    cv2.waitKey(0)


if __name__ == "__main__":
    main()