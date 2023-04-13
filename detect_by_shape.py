from imutils import contours
from skimage import measure
import argparse
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw
import hungarian

input_file = '3.mp4'
max_area = 120
min_area = 25

#izdvajanje pozadine od muva
def get_background(file_path):
    cap = cv2.VideoCapture(file_path)
    # random izvlacenje 200 frejmova i racunanje sredine 
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)
    # cuvanje frejmova u niz
    frames = []
    for idx in frame_indices:
        # frame id postaje bas taj frejm
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    # racunanje svrednje vrednosti
    median_frame = np.median(frames, axis=0).astype(np.uint8)

    return median_frame

# vracamo pozadinu snimka koji ucitavamo
background = get_background(input_file)
# konvertovanje pozadine u grayscale format 
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

circle_center = (len(background[0])//2, len(background)//2)
circle_radius = len(background)//2


cap = cv2.VideoCapture(input_file)
# visina i sirina video frejma
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"outputs/{input_file.split('/')[-1]}"
out = cv2.VideoWriter(
    save_name,
    cv2.VideoWriter_fourcc(*'mp4v'), 10,
    (frame_width, frame_height)
)

while (cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_diff = np.invert(cv2.absdiff(gray, background))
        
        # Perform adaptive thresholding to create a binary image
        thresh = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 15)

        # Remove small noise using morphology opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours in the image
        contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on size and shape (must be roughly elliptical)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max_area and area > min_area:
                ellipse = cv2.fitEllipse(cnt)
                (ex, ey), (ma, Mi), angle = ellipse
                dist = np.sqrt((ex - circle_center[0])**2 + (ey - circle_center[1])**2)
                if dist < circle_radius:
                    cv2.ellipse(img, ellipse, (0, 255, 0), 2)

        cv2.circle(img, circle_center, circle_radius, (0, 0, 255), 2)
        # Display the results
        imS = cv2.resize(img, (frame_width//2, frame_height//2))
        cv2.imshow('Original Image', imS)
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

       

         
    
