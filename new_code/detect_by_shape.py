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

# get the number of flies from user
num_flies = int(input("Enter the number of flies in the video: "))

# initialize array to store previous positions of all flies
prev_positions = np.zeros((num_flies, 2))

# return the background of the loaded video
background = gb.get_background(input_file)

# position of the limiting circle
circle_center = (len(background[0])//2, len(background)//2)
circle_radius = len(background)//2

cap = cv2.VideoCapture(input_file)

# width and height of the displayed video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
previous_positions = []

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

        # array to store the current positions of all flies
        curr_positions = np.zeros((num_flies, 2))

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
                    
                    # find the index of the fly whose previous position is closest to the current position of the ellipse
                    min_dist = np.inf
                    index = -1
                    for i in range(num_flies):
                        if prev_positions[i, 0] == 0 and prev_positions[i, 1] == 0:
                            # if there's no previous position available for this fly, use it as current position
                            curr_positions[i, :] = [ex, ey]
                            index = i
                            break
                        else:
                            dist = np.sqrt((prev_positions[i, 0] - ex)**2 + (prev_positions[i, 1] - ey)**2)
                            if dist < min_dist:
                                min_dist = dist
                                index = i
                    # assign the current position of the ellipse to the closest fly
                    curr_positions[index, :] = [ex, ey]

                    # label the ellipse with the fly number
                    cv2.putText(img, str(index+1), (int(ex), int(ey)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.ellipse(img, ellipse, (0, 255, 0), 2)

        # Display the limiting circle
        cv2.circle(img, circle_center, circle_radius, (0, 0, 255), 2)

        # Save current positions of all flies
        current_positions = previous_positions
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max_area and area > min_area:
                # get ellipse parameters
                ellipse = cv2.fitEllipse(cnt)
                (ex, ey), (ma, Mi), angle = ellipse

                # use only ellipses that are located inside the limiting circle
                dist = np.sqrt((ex - circle_center[0])**2 + (ey - circle_center[1])**2)
                if dist < circle_radius:
                    min_i = 0
                    min_d = dist = np.sqrt((prev_positions[0, 0] - ex)**2 + (prev_positions[0, 1] - ey)**2)
                    for i in range(num_flies):    
                        dist = np.sqrt((prev_positions[i, 0] - ex)**2 + (prev_positions[i, 1] - ey)**2)
                        if(dist < min_d):
                            min_d = dist
                            min_i = i

                    # Save the center of the detected ellipse as the position of the fly
                    current_positions[min_i][0]=ex
                    current_positions[min_i][1]=ey
                    
                    # Label the flies on the image by number
                    cv2.putText(img, str(min_i), (int(ex), int(ey)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Save previous positions of all flies
        if len(previous_positions) == 0:
            # First iteration: fill previous positions with 0s for missing flies
            for i in range(num_flies):
                if i < len(current_positions):
                    previous_positions.append(current_positions[i])
                else:
                    previous_positions.append([0, 0])
        else:
            # Update previous positions of all flies
            new_previous_positions = []
            for i in range(num_flies):
                # Check if current position of the fly is found in the current frame
                if i < len(current_positions):
                    curr_pos = current_positions[i]

                    # Check if the fly was detected as a part of a larger object in the previous frame
                    match_found = False
                    for j in range(num_flies):
                        if np.linalg.norm(np.array(curr_pos) - np.array(previous_positions[j])) < 20:
                            # Match found: this fly is in the same position as the fly in the previous frame
                            new_previous_positions.append(curr_pos)
                            match_found = True
                            break
                    if not match_found:
                        # Match not found: estimate position of the fly based on its previous position and its movement
                        prev_pos = previous_positions[i]
                        dx = curr_pos[0] - prev_pos[0]
                        dy = curr_pos[1] - prev_pos[1]
                        est_pos = [curr_pos[0] + dx, curr_pos[1] + dy]
                        new_previous_positions.append(est_pos)
                else:
                    # Fly not detected in current frame: use its previous position
                    new_previous_positions.append(previous_positions[i])

            # Update previous_positions with new_previous_positions
            previous_positions = new_previous_positions

        # Display the limiting circle
        cv2.circle(img, circle_center, circle_radius, (0, 0, 255), 2)

        # Display the results
        imS = cv2.resize(img, (frame_width//2, frame_height//2))
        cv2.imshow('Original Image', imS)
        cv2.waitKey(1)

    else:
        # End of the video
        break

cap.release()
cv2.destroyAllWindows()