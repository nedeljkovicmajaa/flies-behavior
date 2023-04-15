from imutils import contours
import cv2
import matplotlib.pyplot as plt
import numpy as np


def image_processing(img, background):

# stara obrada - ali adaptivni treshold pojede muve

    # convert frame to gray foramt 
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # exclude background from the frame
    #frame_diff = np.invert(cv2.absdiff(gray, background))
    
    # Perform adaptive thresholding to create a binary image
    #thresh = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

    # Remove small noise using morphology opening
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)


#ovo pojede puno muve ali svakako detektuje sve. nekad detektuje pozadinu - uvesti kruznicu u kodu defined_ellipse

    # konvertovanje frejma u grayscale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # razlika trenutnog i baznog frejma 
    frame_diff = np.invert(cv2.absdiff(gray, background))

    kernel = np.ones((2, 2), np.uint8)
    # tresholdovanje da prebacimo frejm u binarni
    thresh = ~cv2.threshold(frame_diff, 180, 255, cv2.THRESH_BINARY)[1]
    #img_erosion = ~cv2.erode(~thresh, kernel, iterations=5)
    #thresh = ~(cv2.dilate(img_erosion, kernel, iterations=5))
    thresh = cv2.erode(thresh, kernel, iterations = 1)

    return thresh