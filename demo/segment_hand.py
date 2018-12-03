import numpy as np
import cv2
import os

#boundaries here represent RGB values
#edit them to match the color pixel values of 
#the gloves you're using
boundaries = [
    ([0, 120, 0], [140, 255, 100]),
    ([25, 0, 75], [180, 38, 255])
]

def hand_segment(frame):

    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask1 = cv2.inRange(frame, lower, upper)

    lower, upper = boundaries[1]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask2 = cv2.inRange(frame, lower, upper)

    mask = cv2.bitwise_or(mask1, mask2)
    output = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    return gray

def save_seg_frames(direct, new_direct):
    """
    given a directory to some frames
    will segment out the hands
    in the color range represented by boundaries
    and save them in new_direct
    """
    frames = os.listdir(direct)
    for frame in frames:

       image = cv2.imread(direct + frame)
       seg_frame = hand_segment(image)
       cv2.imwrite(new_direct + frame, seg_frame)

    print("Finished segmenting hands")

