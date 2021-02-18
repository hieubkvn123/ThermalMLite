import cv2
from .detect_utils import detect_and_align

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

cam = cv2.VideoCapture(0)
while(True):
    ret, frame = cam.read()
    if(ret):
        images, locations = detect_and_align(frame)

        if(len(images) > 0):
            cv2.imshow('Face', images[0])

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1)
        if(key == ord('q')):
            break

cam.release()
cv2.destroyAllWindows()
