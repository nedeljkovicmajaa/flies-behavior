import cv2
import numpy as np

def get_background(file_path):

    cap = cv2.VideoCapture(file_path)

    # get random 200 frames and count average
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)

    # save all frames in array
    frames = []
    for idx in frame_indices:
        # frame id is becoming exactly that frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    # average
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    # background to grayscale format 
    median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)

    return median_frame