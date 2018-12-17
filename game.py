import numpy as np
import gym
from model import DQN
from preprocess import stack_frames
from collections import deque

class Game(object):
	""" Create the game. Train a model for it and simulate
    Parameters
    ------------
    gameName : string (default: 'Pong-v4')
        Name of the game env in OpenAI Gym
	total_episodes : int
		Total number of episodes of a game to play
	train_or_test : int (default: 2)
		Should I train (1) the network or test it(2)
	doLoadNetwork : Boolean (default: True)
		Should I load the network
	"""
	
	def __init__ (self,gameName,total_episodes,train_or_test,doLoadNetwork=True):
		
		self.createGame(gameName)		
		self.TOT_EPISODES = total_episodes #no. of episodes
		self.MAX_STEPS =  2000 #max steps taken every episode
		
		self.stack_size = 4
		self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(self.stack_size)], maxlen=4)
		
		self.dqn = DQN(self.action_space)
		
		self.path = "saved.h5"
		
		if doLoadNetwork:
			self.load_network()
		
		if train_or_test == 1:
			self.trainAgent()
		elif train_or_test == 2:
			self.load_network()
			self.simulate()
	
	def load_network(self):
		self.dqn.load_network(self.path)
		
	def createGame(self,gameName):
		self.env = gym.make(gameName)
		self.action_space = self.env.action_space.n
		self.env.reset()
		#self.env.render()
		
	def trainAgent(self):
	
		for episode in range(self.TOT_EPISODES):

			state = self.env.reset()
			#self.env.render()
			rewards = 0
			#preprocess and stack the frame/state.
			state, self.stacked_frames = stack_frames(self.stacked_frames, state, True)
			
			for step in range(self.MAX_STEPS):
				if (step%100==0):
					print("Episode No.: ", episode, "Step No.: ", step)
				#ACT
				action = self.dqn.predictAction(state)
				#decay epsilon
				self.dqn.decayEpsilon(episode)
				#Perform the action and get the next_state, reward, and done information
				next_state, reward, done, _ = self.env.step(action)
				#append this state to the frame. Pass the previous stacked frame.
				next_state, self.stacked_frames = stack_frames(self.stacked_frames, next_state, False)
				rewards+=reward
				
				#add experience into memory
				self.dqn.remember(state, action, reward, next_state, done)
				
				state = next_state
				
				if done:
					print("took %d steps" %step)
					print("Earned a total of reward equal to ", rewards)
					break
			
			# TRAIN
			self.dqn.replay()
			self.dqn.target_train()
			
			# Save the network every 1000 iterations
			if observation_num % 10 == 9:
				print("Saving Network")
				self.dqn.save_network(self.path)
		self.endEnvironment()
		

	def simulate(self):
	
		state = self.env.reset()
		self.env.render()
		done = False
		tot_reward = 0
		
		while not done:
			state,_ = stack_frames(self.stacked_frames, state, True)
			action = np.argmax(self.dqn.getModel().predict(state)[0])		##################DOES TGIS WORK>>????????/
			new_state,action,reward,done = self.env.step(action)
			self.env.render()
			state = new_state
			tot_reward+=reward
		
		print(tot_reward)
		self.endEnvironment()
	
	def endEnvironment(self):
		self.env.env.close()

"""
COMMENTS!
ALL HYPERPARAMS SET KARO!!
"""

if __name__ == '__main__' :
	spaceInvader = Game('SpaceInvaders-v0',1,1,False)