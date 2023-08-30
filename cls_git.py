import cv2
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Paths, Tracker, Video
from datetime import datetime




class Yolo_detections:
    '''Class with results of detecting people with YOLOv8 model'''

    def __init__(self, path):
        # we use tensorrt speeded-up weights
        self.model = YOLO(path)

    def detect(self, frame):
        '''Detecting people'''
        yolo_detections = self.model.predict(
            frame, classes=[0], conf=0.4, verbose=False)
        res = yolo_detections[0].boxes.cpu().numpy()
        boxes = res.xyxy.astype(np.uint32)
        cls = res.cls.astype(np.uint8)
        conf = res.conf
        return boxes, cls, conf


class Norfair_Detections:
    '''Norfair is used as a tracker standard in our company'''

    def __init__(self):
        self.tracker = Tracker(
            distance_function="euclidean", distance_threshold=300)

    def transform_yolo2norfair(self, yolo):
        '''Pass the result of yolo detections for Norfair Tracker '''
        self.boxes, self.cls, self.conf = yolo
        detections = []

        for i, box in enumerate(self.boxes):
            detections.append(
                [box[0], box[1], box[2], box[3], self.conf[i], self.cls[i]])
        detections = np.asarray(detections)
        norfair_detections = [Detection(points) for points in detections]

        return norfair_detections

    def update(self, yolo_det):
        '''The function that updates tracking results in the main loop '''

        norfair_detections = self.transform_yolo2norfair(yolo_det)
        tracked_objects = self.tracker.update(detections=norfair_detections)
        return tracked_objects

    def draw_bboxes(self, frame, res):
        '''The function that draws bounding boxes on people '''

        for box in res:
            x1, y1 = int(box.estimate[0, 0]), int(box.estimate[0, 1])
            x2, y2 = int(box.estimate[0, 2]), int(box.estimate[0, 3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)



def get_last_pixel_value(image):
    '''The fucntion that checks whether the image is corrupted or not '''
    height, width, _ = image.shape

    last_pixel = image[height - 1, width - 1]
    blue = last_pixel[0]
    green = last_pixel[1]
    red = last_pixel[2]
    if (blue in range(123, 130)) and (green in range(123, 130)) and (red in range(123, 130)):
        return True
    else:

        return False

