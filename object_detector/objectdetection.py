import os, sys
import cv2
import numpy as np
import tensorflow as tf
import argparse
import collections

sys.path.append('documentdetection/slim')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def get_roi_from_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood']
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)

  classes_name = []
  scores_value = []
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_color_map[box] = groundtruth_box_visualization_color
      else:
        display_str = ''
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
              classes_name.append(class_name)
              scores_value.append(int(100*scores[i]))
            else:
              class_name = 'N/A'
            display_str = str(class_name)
        if not skip_scores:
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
        box_to_display_str_map[box].append(display_str)
        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  rois = []
  lst_boxes = []
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box
    ymin, xmin, ymax, xmax = int(ymin*image.shape[0]), int(xmin*image.shape[1]), int(ymax*image.shape[0]), int(xmax*image.shape[1]) 
    roi = image[ymin:ymax, xmin:xmax]

    rois.append(roi)
    lst_boxes.append([ymin, xmin, ymax, xmax])
    

  return rois, classes_name, scores_value, lst_boxes


def detect_object(sess, img_expanded, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):
    
    print("[INFO] Performing detection...")
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: img_expanded})

    return boxes, classes, scores, num


def save_result_rois(output_path, image, boxes, classes, scores, category_index):
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    rois, classes, scores, lst_box = get_roi_from_image_array( image,
                                    np.squeeze(boxes),
                                    np.squeeze(classes).astype(np.int32),
                                    np.squeeze(scores),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=8,
                                    min_score_thresh=0.5)
    
    for i, roi in enumerate(rois):
        with open('rois.txt', 'a') as f:
            f.write("{}, {}\n".format( classes[i], lst_box[i]))

        cv2.imwrite(output_path + os.path.sep + classes[i] + '_' + str(scores[i]) + ".png", roi)


def return_result_rois(image, boxes, classes, scores, category_index):
    
    rois, classes, scores, lst_box = get_roi_from_image_array( image,
                                    np.squeeze(boxes),
                                    np.squeeze(classes).astype(np.int32),
                                    np.squeeze(scores),
                                    category_index,
                                    use_normalized_coordinates=True,
                                    line_thickness=8,
                                    min_score_thresh=0.1)
    
    return classes, scores, lst_box


def draw_detected_objects(img, boxes, classes, scores, category_index, max_only=True):
    if max_only:
      max_idx = np.argmax(scores[0])
      max_score = scores[0][max_idx]
      max_box = boxes[0][max_idx]
      max_class = classes[0][max_idx]

      # print("[INFO] Drawing results...")
      # Draw the results of the detection (aka 'visulaize the results')
      vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        np.array([max_box]),
        np.array([max_class]).astype(np.int32),
        np.array([max_score]),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.10)

    else:
      max_score = scores[0]
      max_box = boxes[0]
      max_class = classes[0]

      vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        np.squeeze(np.array([max_box])),
        np.squeeze(np.array([max_class]).astype(np.int32)),
        np.squeeze(np.array([max_score])),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.10)

    return max_score, max_box, max_class



def init_model(model_name):
    # OBJ_DETECTION_DATA = 'data\\object_detection_data\\'
    OBJ_DETECTION_DATA = 'C:\\Users\\Valberto\\Workspace\\audiauto\\data\\trained_models'
    INFERENCE_NAME = 'inference_graph_cpf'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, OBJ_DETECTION_DATA, INFERENCE_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, OBJ_DETECTION_DATA, INFERENCE_NAME, 'labelmap.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    print("[INFO] Loading Tensorflow model into memory...")
    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    return sess, detection_graph, category_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default='data\\sample_data\\241474_p4_RG-F.png')
    ap.add_argument('-ig', '--inference_graph', default='')
    args = vars(ap.parse_args())
    
    sess, detection_graph, category_index = init_model()

    # Define input and output tensors (i.e. data) for the object detection classifier
    print("[INFO] Retrieving paramaters from detection graph...")
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    img = cv2.imread(args['input'])
    img_expanded = np.expand_dims(img, axis=0)
    print(img.shape, img_expanded.shape)

    boxes, classes, scores, num = detect_object(sess, img_expanded, 
                                                image_tensor, 
                                                detection_boxes,
                                                detection_scores,
                                                detection_classes, 
                                                num_detections)
    draw_detected_objects(img, boxes, classes, scores, category_index)


if __name__ == '__main__':
    main()