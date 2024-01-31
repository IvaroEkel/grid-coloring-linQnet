# Plan/layout for the project

The idea is to use a similar structure agent-game-model as used by in the sng-ai game (view sng_files folder) to set up our game as controlled by an ANN

to make the choices on how to color a given cell in the board at every step of the game

# Game

The script game.py describing the game should contain the following features:

- A class indicating the possible next moves, inheriting Eum class.
- The class defining the game and its methods
- A comprehensive set of variables to efficiently define the state of the game in a way that
  can be passed to the NN

```python

```


## Agent

In this case, the state of the game and the logic will be a bit more complicated than that for the snake game. 

The agent controlling the snake only had one type of decision (with three possibilities) to make: 

- Where to turn next? Keep straight, turn left or turn right, encoded as [left, straight, right]

Now the agent painting the cells has two kinds of decisions to make:

- Where to move
- What color to paint the cell
