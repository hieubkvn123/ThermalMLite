import os
import cv2, dlib
import numpy as np
from .utils import extract_left_eye_center, extract_right_eye_center, get_rotation_matrix, crop_image
from .detect import detect_faces

base_path = os.path.dirname(os.path.realpath(__file__))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(base_path, "shape_predictor_68_face_landmarks.dat"))

def detect_and_align(img, scale=1):    
    height, width = img.shape[:2]
    s_height, s_width = height // scale, width // scale
    img = cv2.resize(img, (s_width, s_height))

    # dets = detector(img, 1)
    dets = detect_faces(img)

    faces = []
    locations = []

    for i, detection in enumerate(dets):
        left, top, right, bottom = detection

        rect = dlib.rectangle(left, top, right, bottom)
        shape = predictor(img, rect)
        left_eye = extract_left_eye_center(shape)
        right_eye = extract_right_eye_center(shape)

        M = get_rotation_matrix(left_eye, right_eye)
        rotated = cv2.warpAffine(img, M, (s_width, s_height), flags=cv2.INTER_CUBIC)

        cropped = crop_image(rotated, rect)

        faces.append(cropped)
        locations.append((rect.left(), rect.top(), rect.right(), rect.bottom()))

    return faces, locations
