import gym
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, Dense, Activation, Flatten
from keras.optimizers import Adam
from collections import deque

class DDQN(object):
	"""The Model implementing Deep Q Learning."""
	
	def __init__(self, num_actions, input_shape=(84, 84, 4)): 
		""" Initialize the model, create its architecture.
		Parameters
		------------
		num_actions : int 
			length of the action_space of the environment
		input_shape : (int,int,int) (default: (84,84,4))
			The shape of the state. Stack of 4 states of 84X84 pixels.		
		"""
		
		### Model Hyperparameters
		self.INPUT_STATE_SIZE = input_shape # frame is 84x84 with 4 channels 
											# (since stacking 4 frames)
		self.NUM_ACTIONS = num_actions		# action_space.n of env
		### Exploration Hyperparameters
		self.epsilon = 1					# determines if agent should 
											# explore (take a random action) or 
											# exploit (use the model to take a step)											#
		self.min_epsilon = 0.01				
		self.max_epsilon = 1
		self.decay_rate = 0.00001			# to reduce exploration with time.
		### Q Learning Hyperparameters
		self.learning_rate = 0.00001 		# eta. change to 0,00025(for pong)
		self.gamma = 0.99					# discount rate - to affect future rewards.
		self.tau = 0.125					# for target model. 	0,01 (?)
		### Memory Parameters
		self.memory_size = 1000000
		# experience replay/buffer. Store each experience in this object.
		self.memory = deque(maxlen=self.memory_size)
		
		############ CREATING THE DDQN MODEL ARCHITECTURE
		self.constructArchitecture()	 
		
	def constructArchitecture(self):
		"""Create the DDQN model architecture defined my the DeepMind paper."""		
		# input: (84,84,3). Output - num of actions. 
		# 3 convolutional layers. 2 Dense (fully connected) layers
		
		self.model = Sequential()
		self.model.add(Convolution2D(32, (8, 8), 
						input_shape=self.INPUT_STATE_SIZE, strides=(4, 4)))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
		self.model.add(Activation('relu'))
		self.model.add(Convolution2D(64, (3, 3)))
		self.model.add(Activation('relu'))
		self.model.add(Flatten())
		self.model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.model.add(Dense(self.NUM_ACTIONS))
		self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		# Create the target network model.
		self.target_model = Sequential()
		self.target_model.add(Convolution2D(32, (8, 8), 
							input_shape=self.INPUT_STATE_SIZE, strides=(4, 4)))
		self.target_model.add(Activation('relu'))
		self.target_model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
		self.target_model.add(Activation('relu'))
		self.target_model.add(Convolution2D(64, (3, 3)))
		self.target_model.add(Activation('relu'))
		self.target_model.add(Flatten())
		self.target_model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.target_model.add(Dense(self.NUM_ACTIONS))
		self.target_model.compile(loss='mse', 
								optimizer=Adam(lr=self.learning_rate))
		self.target_model.set_weights(self.model.get_weights())
	
	def predictAction(self, state):
		"""Should the agent explore or exploit?"""
		
		""" Pick a random decimal. 
		If lesser than epsilon -> explore -> pick a random action.
		Else -> explot -> use the model to predict action.		
		"""
		if np.random.rand() <= self.epsilon:
			# The agent acts randomly
			action = np.random.randint(0, self.NUM_ACTIONS)
			return action
		#else - Predict the reward value based on the given state
		#keras conv2d expects a 4D input. So add an empty axis. 
		state = np.expand_dims(state, axis=0)
		action = np.argmax(self.model.predict(state)[0])    
		return action    
	
	def remember(self, state, action, reward, next_state, done):
		"""Add an experience to memory"""
		self.memory.append((state, action, reward, next_state, done))
	
	def replay(self, batch_size=32):
		""" Expereince Replay. 
		Train the agent based on the experiences stored in memory.
		Train in batches of 32."""
		
		# empty memory problem:
		if len(self.memory) < batch_size: 
			return
		
		# sample 32 experiences from memory.
		minibatch = random.sample(self.memory, batch_size)
		
		for state, action, reward, next_state, done in minibatch:
			target = self.model.predict(state)[0]		
			# predict action from target network
			fut_action = self.target_model.predict(next_state)[0]	
			# Map reward for the action undertaken.
			target[action] = reward
			
			if not done:
				#Update reward: Q(s,a) = r + gamma * argmax-a Q(s',a')
				target[action] = reward + (self.gamma * np.argmax(fut_action))
			#Train model on updated targets.
			self.model.fit(state, target, epochs=1, verbose=0)
		
		
	def decayEpsilon(self):
		"""Decay epsilon so that the agent exploits more with time."""
		if(self.epsilon>self.min_epsilon):
			# Reduce epsilon (because we need less and less exploration)
			self.epsilon = self.min_epsilon \
						+ (self.max_epsilon - self.min_epsilon) \
						* np.exp(-self.decay_rate*self.decay_rate)

	def target_train(self):
		"""Sunc model and tager model weights."""
		model_weights = self.model.get_weights()
		target_model_weights = self.target_model.get_weights()
		for i in range(len(model_weights)):
			target_model_weights[i] = self.tau * model_weights[i] \
						+ (1 - self.tau) * target_model_weights[i]
						
		self.target_model.set_weights(target_model_weights)
		
	def save_network(self, path):
		"""Saves model at specified path as h5 file"""
		self.model.save(path)
		print("Successfully saved network.")
		
	def load_network(self, path):
		print("Loading the saved model from", path)
		self.model = load_model(path)
		print("Successfully loaded network.")
		
	def getModel(self):
		return self.model
		
#if __name__ == "__main__":
#	spaceInvader = Game('SpaceInvaders-v0',1000,1,False)