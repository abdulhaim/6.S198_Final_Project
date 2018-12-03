import cv2
import os

from segment_hand import *

def segment_video(vid, direct):
	"""
	Given a video, segments it into frames
	and stores in 'direct'
	"""
	vidcap = cv2.VideoCapture(vid)
	success, image = vidcap.read()
	count = 0
	while success:
		segmented_image = hand_segment(image)
		cv2.imwrite(direct + "frame%d.jpg" % count, segmented_image)
		success,image = vidcap.read()
		count += 1
		
	print('Finished Segmenting: ' + vid)



