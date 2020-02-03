import os, sys
import argparse
import cv2
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'object_detector')))

from object_detector.detector import load_graph, detect_object
from yolo_detector.yolo import YoloDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input',        default='videos\\car_chase_01.mp4',
                                     help='Input training path containing training images of a single class')
    ap.add_argument('-a', '--arch',    default='yolo',
                                     help='What model architecture to use for detection (yolo | faster_rcnn | ssd_mobilenet)')
    ap.add_argument('-if', '--inf_input',    default='object_detector\\models\\ssd_mobilenet_v2_coco_2018_03_29',
                                     help='Inference graph directory in case yolo not selected')

    args = vars(ap.parse_args())

    # initialization phase
    if args['arch'] == 'yolo':
        detector = YoloDetector()
    elif args['arch'] == 'faster_rcnn' or args['arch'] == 'ssd_mobilenet':
        detector, sess, categories = load_graph(args['inf_input'])
    else:
        print('[ERROR] Invalid Architecture option.')
        return

    cap = cv2.VideoCapture(args['input'])

    if not cap.isOpened():
        print('[ERROR] Invalid video.')
        return

    # loop phase
    while (1):
        ret, frame = cap.read()

        # restart video stream when it reaches the end
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Perform detection
        start = time.time()
        if args['arch'] == 'yolo':
            boxes, confidences, classes = detector.detect(frame)
            result = detector.draw_detection(frame, boxes, confidences, classes)
        else:
            result, _, _, _ = detect_object(detector, sess, categories, frame)

        end = time.time()
        fps = 1 / (end - start)
        
        # writing fps on image
        text = "fps: {:.3f}".format(fps)
        cv2.putText(result, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0,0,255), 3)
        cv2.imshow('result', result)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
    cv2.destroyAllWindows()
    cap.release()
        

if __name__ == "__main__":
    main()