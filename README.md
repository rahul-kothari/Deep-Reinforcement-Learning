# Deep Reinforcement Learning 
## DDQN (Double Deep Q Networks)
## Atari 2600 - Space Inavders

Please refer to the documentation to understand the code and the theory.

### Files:

requirements.txt - contains the list of libraries installed.

main.py – simply run this and follow the instructions to simulate the game SpaceInvaders-v0

game.py – this file loads the environment, trains the agent and simulates the game.

model.py – contains the code for running the DDQN (Double Deep Q Network) architecture.

preprocess.py – This file contains methods for preprocessing the frames of the video games (downscaling and stacking frames).

playingSpaceInvaders.mp4 – A video of the trained agent playing the game.

NOTE - Although the code was written for SpaceInvaders-v0 Atari Game, you can use the exact same code for any other atari game
(such as `'Pong-v4', 'Breakout-v0'` etc.).  

Also note that I haven't provided my trained network, so you will have to first train your model.