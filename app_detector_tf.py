import os, sys, cv2
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'object_detector')))

import argparse
from object_detector.detector import load_graph, detect_object_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',        default='images\\traffic.jpeg',
                                     help='Input training path containing training images of a single class')
    ap.add_argument('-o', '--inf_input',    default='object_detector\\models\\faster_rcnn_inception_v2_coco_2018_01_28',
                                     help='Output path of the trained model inference graph.')

    args = vars(ap.parse_args())

    detection_graph, sess, categories = load_graph(args['inf_input'])
    results = detect_object_path(detection_graph, sess, categories, args['input'])

    cv2.imshow('img', results[0])
    cv2.waitKey(0)

    
if __name__ == "__main__":
    main()