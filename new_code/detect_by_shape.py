from imutils import contours
import cv2
import matplotlib.pyplot as plt
import numpy as np

import get_background as gb
import image_processing as imp

# video path
input_file = '../3.mp4'

# min and max areas of ellipse
max_area = 120
min_area = 25

# return the background of the loaded video
background = gb.get_background(input_file)

# position of the limiting circle
circle_center = (len(background[0])//2, len(background)//2)
circle_radius = len(background)//2

cap = cv2.VideoCapture(input_file)
# width and height of the displayed video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# iterating through all frames
while (cap.isOpened()):

    # reading current frame
    ret, img = cap.read()

    # if the frame exists
    if ret == True:
        
        # preprocessign of the frame
        opening = imp.image_processing(img, background)

        # Find contours in the image
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on size and shape (must be roughly elliptical)
        for cnt in contours:

            # calculate the area of the current ellipse
            area = cv2.contourArea(cnt)

            # use only ellipses that fit certain size 
            if area < max_area and area > min_area:

                # get ellipse parameters
                ellipse = cv2.fitEllipse(cnt)
                (ex, ey), (ma, Mi), angle = ellipse

                # use only ellipses that are located inside the limiting circle
                dist = np.sqrt((ex - circle_center[0])**2 + (ey - circle_center[1])**2)
                if dist < circle_radius:
                    
                    # display the ellipse
                    cv2.ellipse(img, ellipse, (0, 255, 0), 2)

        # Display the limiting circle
        cv2.circle(img, circle_center, circle_radius, (0, 0, 255), 2)

        # Display the results
        imS = cv2.resize(img, (frame_width//2, frame_height//2))
        cv2.imshow('Original Image', imS)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

       

         
    
