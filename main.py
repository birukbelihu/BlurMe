import time
from collections import deque

import cv2
import numpy as np
from blurme_utils import blur_face
from constants import *

previous_frame_time = 0
fps_history = deque(maxlen=10)

net = cv2.dnn.readNetFromCaffe(get_prototxt_file(), get_caffe_model())

video_capture = cv2.VideoCapture(1)

while video_capture.isOpened():
    is_successful, frame = video_capture.read()
    if not is_successful:
        break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - previous_frame_time + 1e-5)
    previous_frame_time = new_frame_time

    fps_history.append(fps)
    average_fps = sum(fps_history) / len(fps_history)
    fps_text = f"FPS: {int(average_fps)}"

    cv2.putText(frame, fps_text, (10, 25), cv2.QT_FONT_NORMAL,
                0.7, (219, 109, 24), 2)

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_bounding_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype(int)
            face_bounding_boxes.append((x1, y1, x2, y2))

            frame = blur_face(frame, face_bounding_boxes)

            # If You Want To Draw A Bounding Box Around The Blurred Face You Can Uncomment This Line
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow(get_app_name(), frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or chr(key) in exit_keys():
        print(f"Exiting {get_app_name()}...")
        break

video_capture.release()
cv2.destroyAllWindows()
