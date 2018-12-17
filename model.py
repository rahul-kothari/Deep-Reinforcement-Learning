import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Activation, Flatten
from keras.optimizers import Adam
from collections import deque
"""
### MODEL HYPERPARAMETERS
INPUT_STATE_SIZE = (84, 84, 4)      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels) 
NUM_ACTIONS = env.action_space.n 	 # 8 possible actions


### TRAINING HYPERPARAMETERS
total_episodes = 50            # Total episodes for training
max_steps = 500              # Max possible steps in an episode
batch_size = 32                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate
learning_rate =  0.00001      # Alpha (aka learning rate)

### MEMORY HYPERPARAMETERS
memory_size = 1000000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked
"""

class DQN(object):
	def __init__(self, num_actions, input_shape=(84, 84, 4)):
	
		self.INPUT_STATE_SIZE = input_shape  # input to CNN is a stack of 4 frames of 84x84. 
		self.NUM_ACTIONS = num_actions
		self.epsilon = 1
		self.min_epsilon = 0.01
		self.max_epsilon = 1
		self.decay_rate = 0.995
		self.learning_rate = 0.00001
		self.gamma = 0.85
		self.tau = 0.125
		
		self.constructArchitecture()
		self.memory = deque(maxlen=20000) #1 million
		
	def constructArchitecture(self):
		self.model = Sequential()
		self.model.add(Convolution2D(32, (8, 8), input_shape=self.INPUT_STATE_SIZE, strides=(4, 4)))
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

		# Creates a target network as described in DeepMind paper
		self.target_model = Sequential()
		self.target_model.add(Convolution2D(32, (8, 8), input_shape=self.INPUT_STATE_SIZE, strides=(4, 4)))
		self.target_model.add(Activation('relu'))
		self.target_model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
		self.target_model.add(Activation('relu'))
		self.target_model.add(Convolution2D(64, (3, 3)))
		self.target_model.add(Activation('relu'))
		self.target_model.add(Flatten())
		self.target_model.add(Dense(512))
		self.model.add(Activation('relu'))
		self.target_model.add(Dense(self.NUM_ACTIONS))
		self.target_model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		self.target_model.set_weights(self.model.get_weights())
		
	def predictAction(self, state):
		if np.random.rand() <= self.epsilon:
			# The agent acts randomly
			action = np.random.randint(0, self.NUM_ACTIONS)
			return action
		#else - Predict the reward value based on the given state
		state = np.expand_dims(state, axis=0)
		action = np.argmax(self.model.predict(state)[0])    ##################DOES TGIS WORK>>????????/
		return action
    	
	def decayEpsilon(self, num_episode):
		if(self.epsilon>self.min_epsilon):
			# Reduce epsilon (because we need less and less exploration)
			self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) \
				  * np.exp(-self.decay_rate*num_episode)
	
	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def replay(self, batch_size=32):
		# empty memory problem:
		if len(self.memory) < batch_size: 
			return
			
		minibatch = random.sample(self.memory, batch_size)

		for state, action, reward, next_state, done in minibatch:
			target = self.model.predict(state)[0]				
			fut_action = self.target_model.predict(next_state)[0]			##################DOES TGIS WORK>>????????/
			target[action] = reward
			
			if not done:
				#Q(s,a) = r + gamma * argmax-a Q(s',a')
				target[action] = reward + (self.gamma * np.argmax(fut_action))
			self.model.fit(state, target, epochs=1, verbose=0)
			
	def target_train(self):
		model_weights = self.model.get_weights()
		target_model_weights = self.target_model.get_weights()
		for i in range(len(model_weights)):
			target_model_weights[i] = self.tau * model_weights[i] \
						+ (1 - self.tau) * target_model_weights[i]
						
		self.target_model.set_weights(target_model_weights)
		
	def save_network(self, path):
        # Saves model at specified path as h5 file
		self.model.save(path)
		print("Successfully saved network.")
		
	def load_network(self, path):
		self.model = load_model(path)
		print("Succesfully loaded network.")
		
	def getModel(self):
		return self.model
		
#if __name__ == "__main__":
#	spaceInvader = Game('SpaceInvaders-v0',1000,1,False)