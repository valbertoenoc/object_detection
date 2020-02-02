import argparse
import cv2

from yolo_detector.yolo import YoloDetector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="images\\soccer.jpg", help="Path to input image")
    
    args = vars(ap.parse_args())

    yd = YoloDetector()

    image = cv2.imread(args['input'])

    boxes, confidences, classes = yd.detect(image)
    result = yd.draw_detection(image, boxes, confidences, classes)
    
    cv2.imshow('result', result)
    cv2.waitKey()

if __name__ == "__main__":
    main()