import numpy as np
import gym
from model import DDQN
from preprocess import stack_frames
from collections import deque

class Game(object):	
	
	def __init__ (self,gameName,total_episodes=50,train_or_test=2):
		""" Create the game, SpaceInvaders. Train or test it.
		Parameters
		------------
		gameName : string (default: 'Pong-v4')
			Name of the game env in OpenAI Gym
		total_episodes : int
			Total number of episodes of a game to play
		train_or_test : int (default: 2)
			Should I train (1) the network or test it(2)		
		"""
		#additional param:- doLoadNetwork=True
	
		self.createGame(gameName)
		
		### Training Hyperparameters
		self.TOT_EPISODES = total_episodes 	#no. of episodes/epochs
		self.MAX_STEPS =  50000 		   	#max steps taken every episode/epoch
		
		### Preprocessing Hyperparameters
		self.stack_size = 4					#stacking 3 frames at once.					
		self.stacked_frames = deque([np.zeros((84,84), dtype=np.int) 
					for i in range(self.stack_size)], maxlen=4)	
					
		### Model
		self.dqn = DDQN(self.action_space)
		
		self.path = "saved.h5"
				
		# train agent or simulate the game.
		if train_or_test == 1:
			self.trainAgent()
		elif train_or_test == 2:
			#load network before simulating.
			self.load_network()
			self.simulate()
	
	def load_network(self):
		"""load the network from saved.h5"""		
		self.dqn.load_network(self.path)
		
	def createGame(self,gameName):
		"""create game env"""
		self.env = gym.make(gameName)
		self.action_space = self.env.action_space.n
		self.env.reset()
		#self.env.render()
		
	def trainAgent(self):
		"""Train the model for a certain no of episodes"""
		for episode in range(self.TOT_EPISODES):
			#reset environment, stacked frames every episode.
			state = self.env.reset()
			rewards = 0
			#preprocess and stack the frame/state.
			state, self.stacked_frames = stack_frames(self.stack_size,
									self.stacked_frames, state, True)
			
			for step in range(self.MAX_STEPS):
			#for every step in episode:
			
				if (step%100==0):
					print("Episode No.: ", episode, "Step No.: ", step)
				
				#agent acts - explores or exploitation of the model
				action = self.dqn.predictAction(state)
				#reduce epsilon for more exploitation later.
				self.dqn.decayEpsilon()
				#Perform the action and get the next_state, reward, and done vals.
				next_state, reward, done, _ = self.env.step(action)
				#append this state to the frame. Pass the previous stacked frame.
				next_state, self.stacked_frames = stack_frames(self.stack_size,
										self.stacked_frames, next_state, False)
				rewards+=reward
				
				#add this experience into memory (experience buffer)
				self.dqn.remember(state, action, reward, next_state, done)
				
				state = next_state
				
				if done:
					print("took %d steps" %step)
					print("Earned a total of reward equal to ", rewards)
					break
			
				# TRAIN
				self.dqn.replay()
				#sync target_model and model weights every 10k steps.
				if step % 10000 == 9999:
					self.dqn.target_train()
			
			# Save the network every 1000 iterations
			if episode % 5 == 4:
				print("Saving Network")
				self.dqn.save_network(self.path)
		

	def simulate(self):
		"""Test the agen and watch it play for one episode."""
		print("##################################")
		print("SIMULATING GAME - SpaceInvaders..")
		print("##################################")
		
		# Play 3 episodes:
		for i in range(3):
			print("Playing Episode %d" % i)
			state = self.env.reset()
			#self.env.render()
			done = False
			tot_reward = 0
			state,_ = stack_frames(self.stack_size,self.stacked_frames, 
										state, True)
			# play until dead.			
			while not done:
				# get the value predicted by the model and perform that action.
				# keras conv2d expects a 4D input. So add an empty axis. 
				state = np.expand_dims(state, axis=0)
				# predict action directly from the saved neural network.
				action = np.argmax(self.dqn.getModel().predict(state)[0])
				# perform that action.
				state, reward, done, _ = self.env.step(action)
				self.env.render()
				state,_ = stack_frames(self.stack_size,self.stacked_frames, 
										state, False)
				tot_reward+=reward
			print("Reward: ", tot_reward)
		self.env.close() # to avoid sys.meta_path error

if __name__ == '__main__' :
	spaceInvader = Game('SpaceInvaders-v0',train_or_test=2)
	