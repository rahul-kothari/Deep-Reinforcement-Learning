#file to PREPROCESS states

import cv2
import numpy as np
from collections import deque
import random


def downscale(frame):
	#grayscale frame. Change it from 210x160x3 to 210x160x1 (RGB=3 channels)
	black_buffer = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
	
	#rescale the frame to 110x84.
	resized = cv2.resize(black_buffer, (84,110 # !! resize takes (columns,rows)
	#normalize frame:
	#normalized = black_buffer/255.0
	
	#crop frame to 84x84
	result = resized[13:97,:]

	return result

def stack_frames(stack_size, stacked_frames, state, is_new_episode):
	""" Downscale frames, and stack them
	Parameters
	------------
	stack_size : int
		Size of stack (no of channels in NN.
	stacked_frames : (int,int,int)
		Current stacked frames (4X84X84)
	state : (int,int,int) (210,160,3)
		State to be downscaled and stacked
	is_new_episode : Boolean 
		is this frame stacked for a new episode		
	Returns
	-----------
	stacked_state : (84,84,4)
		This state goes as the input to the NN model.
	stacked_frames : (4,84,84)
		Updated frame with new state.
	
	
	"""
	# downscale frame
	frame = downscale(state)
	
	if is_new_episode:
		# Clear our stacked_frames. Create new one
		stacked_frames = deque([np.zeros((84,84), dtype=np.int) 
			for i in range(stack_size)], maxlen=4)		
		
		# Because we're in a new episode, copy the same frame 3x
		stacked_frames.append(frame)
		stacked_frames.append(frame)
		stacked_frames.append(frame)
		stacked_frames.append(frame)					
		
		# Stack the frames
		stacked_state = np.stack(stacked_frames, axis=2)
		
	else:
		# Append to deque, automatically removes the oldest frame
		stacked_frames.append(frame)

		# Build the stacked state (first dimension specifies different frames)
		stacked_state = np.stack(stacked_frames, axis=2)
		# shape = 84x84x4
	return stacked_state, stacked_frames