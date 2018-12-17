import cv2
import numpy as np
from collections import deque
import random

#PREPROCESS
def downscale(frame):
	#make it grayscale, BUT no useful information should be lost (i.e. color of ball shouldnt became same as bg)
	grayscaled = np.mean(frame,2,keepdims = False) #cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	#normalize frame:
	normalized = grayscaled/255.0
	#resize 210x160 to 110x84
	resized = cv2.resize(normalized, (84,110),interpolation = cv2.INTER_LINEAR)
	#crop to 84x84
	result = resized[13:97,:]
	return result

def stack_frames(stacked_frames, state, is_new_episode):
    # downscale frame
	frame = downscale(state)
	stack_size=4
	if is_new_episode:
		# Clear our stacked_frames
		stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
		
		# Because we're in a new episode, copy the same frame 4x
		stacked_frames.append(frame)
		stacked_frames.append(frame)
		stacked_frames.append(frame)
		stacked_frames.append(frame)
		
		# Stack the frames
		stacked_state = np.stack(stacked_frames, axis=2)
		
	else:
		# Append frame to deque, automatically removes the oldest frame
		stacked_frames.append(frame)

		# Build the stacked state (first dimension specifies different frames)
		stacked_state = np.stack(stacked_frames, axis=2) 
	return stacked_state, stacked_frames