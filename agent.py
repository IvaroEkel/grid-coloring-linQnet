import torch
import random
import numpy as np
from collections import deque # deque is a list optimized for removing and adding items
from game import GraphPainterAI, Point, NextCell, Color, ColorDict
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 # the size of the deque
BATCH_SIZE = 1000 # mini batch size
LR = 0.001 # learning rate

class Agent:
    ## class Agent initializer ##
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness 
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if MAX_MEMORY is reached. The agent stores the last 100_000 moves in memory
        # the linear Q network takes as input the number of colors 'ncolors' selected in game.py, and the number of possible actions
        # the output is a vector of equal size.
        self.model = Linear_QNet( 3*3 +  3, 128 + 3*3 + 3, 4+3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    ### class methods ###
    # 1. get_state. This method takes an argument, 'game', which is of the class GraphPainterAI in game.py.
    def get_state(self, game): # the state is a vector of size 1+self.ncolors. The first 4 elements of the vector are the 4 possible directions the agent can move to. The last 3 elements are the 3 possible color choices the agent can make.
        # print message: getting state
        print('getting state')
        # first get the current pairings 
        pairings = game.pairings
        pairings_state = [len(game.pairings[color]) == game.ncolors - 1 for color in range(game.ncolors)] # this is a list with boolean values.
        # now get the state of the board as a flattened list of integers: 0 if the cell has not been painted yet (color = BLACK), 1 if the cell has been painted (color != BLACK)
        board_state = [game.board[(i,j)] != Color.BLACK for i in range(game.board_size) for j in range(game.board_size)]
        # join the two lists
        state = pairings_state + board_state
        # convert the boolean values to integers as a np array
        # state = np.array([int(i) for i in state])
        # or
        state = np.array(state, dtype=int)
        # return the state
        return state

    def remember(self, state, action, color_choice, reward, next_state, done): # decision is a tuple containing the action to move and the color_choice made by the agent
        self.memory.append((state, action, color_choice, reward, next_state, done))

    def train_short_memory(self, state, action, color_choice, reward, next_state, done):
        self.trainer.train_step(state, action, color_choice, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # if the memory is full, then sample a mini batch of size BATCH_SIZE from the memory
            mini_sample = random.sample(self.memory, BATCH_SIZE) 
        else: # if the memory is not full yet, then sample the whole memory
            mini_sample = self.memory
        # use the trainer to train the model. First 'unpack' the mini_sample into its components
        states, actions, color_choices, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, color_choices, rewards, next_states, dones)
        # can also be done with a for loop:
        # for state, decision, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, decision, reward, next_state, done)

    def get_action(self, state): 
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games
        final_move = [0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon: # if the random number is less than epsilon, then choose a random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # get action from Q-network
            state0 = torch.tensor(state, dtype=torch.float) # convert state to a tensor
            prediction = self.model(state0) # get the prediction of our model based on the current state
            move = torch.argmax(prediction).item() # get the index of the maximum value of the prediction
            final_move[move] = 1 # set the final move to 1 at the index of the maximum value of the prediction
        # as the game progresses, we will perform less and less random moves and more and more moves based on the prediction of our model
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = GraphPainterAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done: # done is True when the game is over
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if reward > record:
                record = reward
                agent.model.save()

            print('Game', agent.n_games, 'Score', reward, 'Record:', record)

            plot_scores.append(reward)
            total_score += reward
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()