from imutils import contours
import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_processing(img, background):

    # convert frame to gray foramt 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # exclude background from the frame
    frame_diff = np.invert(cv2.absdiff(gray, background))
    
    # Perform adaptive thresholding to create a binary image
    thresh = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

    # Remove small noise using morphology opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return opening