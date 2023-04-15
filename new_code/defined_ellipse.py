from imutils import contours
import cv2
import matplotlib.pyplot as plt
import numpy as np

import get_background as gb
import image_processing as imp

# video path
input_file = '../3.mp4'

# return the background of the loaded video
background = gb.get_background(input_file)

cap = cv2.VideoCapture(input_file)
# width and height of the displayed video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the templates for the ellipse shape
templates = []
for angle in range(0, 181, 10):
    M = cv2.getRotationMatrix2D((25, 25), angle, 1)
    template = cv2.ellipse(np.zeros((50,50), np.uint8), (25,25), (6,3), 0, 0, 360, 255, -1)
    templates.append(cv2.warpAffine(template, M, (50, 50)))

# keep track of detected objects
detected_objs = []

# iterating through all frames
while (cap.isOpened()):

    # reset the detected objects list
    detected_objs = []

    # reading current frame
    ret, img = cap.read()

    # if the frame exists
    if ret == True:
        
        gray = imp.image_processing(img, background)

        for template_idx, template in enumerate(templates):
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.5)
            for pt in zip(*loc[::-1]):
                # check if the detected object has already been found in a previous template
                #detected_objs.append(pt)
                #cv2.ellipse(img, (pt[0]+25, pt[1]+25), (5,5), 0, 0, 360, (0, 255, 0), 2)
                if not any(np.allclose(pt, obj, atol = 12) for obj in detected_objs):
                    detected_objs.append(pt)
                    cv2.ellipse(img, (pt[0]+25, pt[1]+25), (5,5), 0, 0, 360, (0, 255, 0), 2)
        
        print(len(detected_objs))
        # Display the results
        imS = cv2.resize(img, (frame_width//2, frame_height//2))
        cv2.imshow('Original Image', imS)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()