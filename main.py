from game import Game

print("This model plays an Atari Game of your choice!")

gameName = input("Input the Atari Game name of the game you want to train/test (eg: SpaceInvaders-v0): ")
# = SpaceInvaders-v0 OR Pong-v4 OR Breakout-v0 ....

test_or_train = int(input("Do you want to train (enter 1) or test (enter 2) :"))

while(test_or_train != 1 and test_or_train !=2):
	#while you get invalid input, keep asking
	test_or_train = input("Do you want to train (enter 1) or test (enter 2) :")

#Testing automatically loads the saved network.
spaceInvader = Game(gameName,test_or_train)
