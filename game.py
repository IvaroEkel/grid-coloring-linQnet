import pygame 
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()


class NextCell(Enum):
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4

Point = namedtuple('Point', 'x, y') # this is a named tuple. The name is 'Point'. The tuple has two elements, 'x' and 'y'.

# RGB colors 
BLACK = (0,0,0)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
BLUE2 = (0, 100, 255)
GRAY = (128,128,128)
WHITE = (255, 255, 255)


# RGB colors as a class Enum
class Color(Enum):
    BLACK = 0
    BLUE1 = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    PURPLE = 5
    ORANGE = 6
    PINK = 7
    WHITE = 100
    GRAY = 101
    BLUE2 = 102

# we can use the following dictionary to map the colors to their RGB values
# dictionary mapping the colors to their RGB values
ColorDict = {
    Color.BLACK: BLACK,
    Color.BLUE1: BLUE1,
    Color.RED: RED,
    Color.GREEN: GREEN,
    Color.YELLOW: YELLOW,
    Color.PURPLE: PURPLE,
    Color.ORANGE: ORANGE,
    Color.PINK: PINK,
    Color.GRAY: GRAY,
    Color.WHITE: WHITE,
    Color.BLUE2: BLUE2
}

# and we can call the colors by their names, e.g. Color.BLACK, Color.RED, etc. or by their RGB values, e.g. BLACK, RED, etc.
# or by their integer values, e.g. 0, 3, etc.
# calling by integer values is useful for the neural network, since it can only take numbers as input.
# example: ColorDict[Color.BLACK] returns BLACK, which is the RGB value of the color BLACK.
# example by integer value: ColorDict[Color(0)] returns BLACK, which is the RGB value of the color BLACK.

NCELLSW = 2
NCELLSH = 2 
BLOCK_SIZE = 20 # the size is in pixels
BOARD_WIDTH = NCELLSW*BLOCK_SIZE
BOARD_HEIGHT = NCELLSH*BLOCK_SIZE
SPEED = 10

class GraphPainterAI: # it inherits from the class 'object'. This is the same as 'class GraphPainterAI(object):'
    ### class initializer ###
    def __init__(self, w = BOARD_WIDTH, h = BOARD_HEIGHT, nx = NCELLSW, ny = NCELLSH, ncolors = 2):
        self.w = w
        self.h = h
        self.nx = nx
        self.ny = ny
        self.board_size = self.nx * self.ny  # this is the number of cells in the board
        self.ncolors = ncolors
        self.vcolors = [i for i in range(1,self.ncolors,1)] # this is a list of the colors that the agent can choose from, as integers
        # we need a listof tuples {cell_index: color} to keep track of the cells that have been painted. The cell index is obtained as follows:
        # cell_index = (cell.x, cell.y)
        # the color is an integer value and it will be set to BLACK when initialized.
        # self.board = {(i, j): Color.BLACK for i in range(self.board_size) for j in range(self.board_size)} # Color.BLACK is the default color of the board and it returns the integer value 0.
        self.board = {}
        self.pairings = {} # this is a dictionary that tracks the number of colors that have been paired.
        # self.pairings_state = [] 
        # a color 'color1' is considered to be paired with another color 'color2' if they share a common edge, i.e. 
        # if they are adjacent at least once in the board.
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Graph Painter')
        self.clock = pygame.time.Clock()
        self.reset()
    
    ### class methods ###
    # 1. reset. This initializes the game state.
    def reset(self): # it does not take any arguments, since it intended to act only when the condition of game-over or win is met.
        # self.direction = NextCell.NORTH
        self.pairings = {color: [] for color in self.vcolors} # this initializes the dictionary that tracks the number of colors that have been paired.
        # make the painter start in the middle of the board
        self.painter = Point(self.w/2, self.h/2)
        self.painting = [] # this is a list of tuples. Each tuple contains a cell and its color.
        # make a dictionary in which the coordintates are given as Point(x, y) and the color is given as an integer. Use range() to get the coordinates from 0 to the width and height of the board.
        # in steps equal to the block size.
        self.board = {(i, j): Color.BLACK for i in range(0, 2, 1) for j in range(0, 2, 1)} # Color.BLACK is the default color of the board and it returns the integer value 0.
        self._paint_cell() # this method is defined below. It's function is to paint the cell that the painter is currently in.
        # self.painting = [(self.painter, get_cell_color(self.painter))] # the painting is a list of tuples. Each tuple contains a cell and its color.
        self.score = 0
        self.frame_iteration = 0
        # print 
        print(self.pairings)
        # print the initial state of the board
        print(self.board)
        self.board_size


    # 2. play_step. This is the main method of the class. It takes an argument, 'action', which is the action taken by the agent.
    def play_step(self, action):
        self.frame_iteration += 1
        # 2.1. event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # 2.2. move
        self._move(action) # this method is defined below. It's function is to move the painter to 
        # the neighboring cell in the direction specified by the AI painter's action.
        self._paint_cell() # this method is defined below. It's function is to paint the cell that
        
        self.painting.append(self.painter) # this adds the current cell to the list of cells that have been painted.
        # 2.3. check if game over
        game_over = False
        if self.board_full(): # in this case, the game is over and we will give a large negative reward.
            game_over = True
            reward = -100
            return game_over, self.score
        # 2.3.1 check if the game has been won. In that case we will give a large positive reward.
        if self.game_won():
            game_over = True
            reward = 100
            return game_over, self.score

        # 2.3.1 check if the game has been won. The game is won once all of the pairings have been completed.
        # this means that the length of the list of pairings of each color is equal to the number of colors minus one.
        # if this is the case, then the game is won.
        # 2.4. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 2.5. return game over and score
        return game_over, self.score

    def _move(self, action): # this method takes an argument, 'action', 
        # which is the action taken by the agent.
        # [NORTH, EAST, SOUTH, WEST]
        # 0 1 2 3
        # 1. update the painter's position 
        if action == [1,0,0,0]: # if the action is [1,0,0,0], then move north
            self.painter = Point(self.painter.x, self.painter.y - BLOCK_SIZE) # move north. In the videogame, the y-axis is inverted.   
        elif action == [0,1,0,0]: # if the action is [0,1,0,0], then move east
            self.painter = Point(self.painter.x + BLOCK_SIZE, self.painter.y)
        elif action == [0,0,1,0]: # if the action is [0,0,1,0], then move south
            self.painter = Point(self.painter.x, self.painter.y + BLOCK_SIZE)
        elif action == [0,0,0,1]: # if the action is [0,0,0,1], then move west
            self.painter = Point(self.painter.x - BLOCK_SIZE, self.painter.y)
        else: # if the action is not one of the above, then raise an exception
            raise Exception("Invalid action!")
        
    def _paint_cell(self, color_choice = 1):  # choice = 1 means that the cell will be painted with the color BLUE1. 
        # this default value is necessary for the method to work when called without an argument in reset to initialize the game state.
        
        # 1. paint the cell
        pygame.draw.rect(self.display, ColorDict[Color(color_choice)], pygame.Rect(self.painter.x, self.painter.y, BLOCK_SIZE, BLOCK_SIZE))
        # 2. update the cell color in the board as its corresponding integer value in the correpoding position in the tuple in the list self.board
        self.board[(self.painter.x, self.painter.y)] = color_choice
        
        self.painting.append((self.painter, color_choice)) # add the tuple (self.painter, color_choice) to the list of cells that have been painted.

    def get_cell_color(self):
        # 1. get the color of the cell
        cell_color = self.display.get_at((self.painter.x, self.painter.y)) # display.get_at() returns the color of the pixel at the specified position.
        # 2. get the color of the cell as an integer
        cell_color = ColorDict[Color(cell_color)]
        return cell_color

    def update_board(self):
        # check all the cells in the board and update their colors
        for i in range(0, self.w, BLOCK_SIZE):
            for j in range(0, self.h, BLOCK_SIZE):
                # 1. get the color of the cell
                cell_color = self.display.get_at((i, j))
                # 2. get the color of the cell as an integer
                cell_color = ColorDict[Color(cell_color)]
                # 3. update the board
                self.board[(i, j)] = cell_color

    def board_full(self): # if the board is full, then the game is over.
        # check if the board is full. This is simple: if the length of the list painting is equal to the number of cells in the board, then the board is full.
        if len(self.painting) == (self.w//BLOCK_SIZE)*(self.h//BLOCK_SIZE):
            return True
        
    def game_won(self): # if the game is won, then the game is over.
        # check if the game is won. This is simple: if the length of the list of pairings of each color is equal to the number of colors minus one, then the game is won.
        # vector of booleans. Each boolean is True if the length of the list of pairings of the corresponding color is equal to the number of colors minus one.
        won = [len(self.pairings[color]) == self.ncolors - 1 for color in range(self.ncolors)]
        # if all the elements of the vector are True, then the game is won.
        if all(won):
            return True


    def update_pairings(self):
        # check all the cells in the painting and update the pairings dictionary
        # by getting the colors of the neighboring cells of each cell in the painting
        for cell in self.painting: # cell is a tuple (cell, color)
            # 1. get the neighboring cells
            neighboring_cells = []
            neighboring_cells.append((cell[0], cell[1] - BLOCK_SIZE))
            neighboring_cells.append((cell[0] + BLOCK_SIZE, cell[1]))
            neighboring_cells.append((cell[0], cell[1] + BLOCK_SIZE))
            neighboring_cells.append((cell[0] - BLOCK_SIZE, cell[1]))
            # 2. get the colors of the neighboring cells and append them to the pairings dictionary in the corresponding color key: self.pairings[cell[1]], if they are not already in the list of pairings
            for neighboring_cell in neighboring_cells:
                # if the color is not already in the list of pairings, and if the color is not zero (BLACK) or itself, then append it to the list of pairings
                if self.board[neighboring_cell] not in self.pairings[cell[1]] and self.board[neighboring_cell] != 0 and self.board[neighboring_cell] != cell[1]:
                    self.pairings[cell[1]].append(self.board[neighboring_cell])

    def _update_ui(self):
        self.display.fill(BLACK)
        # update the board
        self.update_board()
        # draw the painting
        for cell in self.painting:
            pygame.draw.rect(self.display, ColorDict[Color(self.board[cell])], pygame.Rect(cell[0], cell[1], BLOCK_SIZE, BLOCK_SIZE))
        # draw the grid
        # for i in range(0, self.w, BLOCK_SIZE):
        #     pygame.draw.line(self.display, GRAY, (i, 0), (i, self.h))
        # for j in range(0, self.h, BLOCK_SIZE):
        #     pygame.draw.line(self.display, GRAY, (0, j), (self.w, j))
        # draw the painter
        pygame.draw.rect(self.display, RED, pygame.Rect(self.painter.x, self.painter.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip() # this updates the display with the changes made in the previous lines of code.

    # def is_out_of_bounds(self, point):
    #     if point.x < 0 or point.x > self.w or point.y < 0 or point.y > self.h:
    #         return True
    #     else:
    #         return False
