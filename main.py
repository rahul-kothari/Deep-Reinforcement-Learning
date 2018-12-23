from game import Game

print("This model plays Space Inavders v0 Atari Game")

test_or_train = int(input("Do you want to train (enter 1) or test (enter 2) :"))

while(test_or_train != 1 and test_or_train !=2):
	#while you get invalid input, keep asking
	test_or_train = input("Do you want to train (enter 1) or test (enter 2) :")

#Testing automatically loads the saved network.
spaceInvader = Game('SpaceInvaders-v0',test_or_train)
