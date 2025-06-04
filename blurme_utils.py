import cv2


def blur_face(frame, face_boxes, blur_strength=(55, 55), sigma=30):
    frame_copy = frame.copy()

    for (x1, y1, x2, y2) in face_boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)
        frame_roi = frame_copy[y1:y2, x1:x2]
        blurred_frame = cv2.GaussianBlur(frame_roi, blur_strength, sigma)
        frame_copy[y1:y2, x1:x2] = blurred_frame

    return frame_copy
