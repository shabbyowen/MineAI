"""
A complete minesweeper game with difficulty setting
"""

import random
import time
import numpy as np
from itertools import product, starmap

# State definitions for a single square
OPEN = 100          # square is touched/revealed
HIDDEN = 101        # square is untouched
FLAGGED = 102       # square is flagged by player
DEATH_MINE = 103    # square of mine that was triggered
MISFLAGGED = 104    # square not mine that was flagged

# State definitions for variable mine:
# 1-8  -  Number of mines in proximity
# 0    -  Empty space
# -1   -  Has mine in square
_EMPTY_ = 0
_MINE_ = -1

DIFF_BEGINNER = {'width': 8, 'height': 8, 'num_mines': 10, 'desc': 'BEGINNER'}
DIFF_INTERMED = {'width': 16, 'height': 16, 'num_mines': 40, 'desc': 'INTERMEDIATE'}
DIFF_EXPERT = {'width': 30, 'height': 16, 'num_mines': 99, 'desc': 'EXPERT'}


class Minesweeper(object):
    """Master class for a Minesweeper class"""

    class MinesweeperBoard(object):
        """Board class for constructing, accessing and changing of a minefield"""

        def __init__(self, height, width, num_mines, first_click):
            """Initialize and generate a mine field on the board
            :type first_click: [int,int]
            """

            if num_mines >= height * width:
                raise ValueError("You placed more mines than number of grids!")

            self.height = height
            self.width = width
            self.num_mines = num_mines

            # Initializing empty board using 2d np.array of a number indicating mines
            board = np.array([[_EMPTY_] * self.width] * self.height, np.int32)

            # Make sure first click is a empty grid
            first_click_serial = first_click[0] * self.width + first_click[1]
            unavailable = []
            for row,col in self.neighbors(first_click[0],first_click[1]):
                unavailable.append(row*self.width + col)
            #print(unavailable)
            available_squares = list(range(self.width * self.height))
            #print(available_squares)
            for serial in unavailable:
                available_squares.remove(serial)
            serials = random.sample(available_squares, k=self.num_mines)

            # Generate mines on the board and stores their locations
            mine_locs = []
            for coordinate_serial in serials:
                row = coordinate_serial // self.width
                col = coordinate_serial % self.width
                board[row][col] = _MINE_
                mine_locs.append((row, col,))

            self.board = board
            self.mines = mine_locs
            
            # Compute number labels for squares surrounding the mines
            all_locs = list(product(list(range(self.height)), list(range(self.width))))
            for row, col in all_locs:
                if (row, col) not in mine_locs:
                    for nrow, ncol in self.neighbors(row, col):
                        if self.board[nrow][ncol] == _MINE_:
                            self.board[row][col] += 1
        
        def get_pos(self, row, col):
            return self.board[row][col]

        def get_num_mines(self):
            return self.num_mines

        def get_mines(self):
            return self.mines

        def get_board(self):
            return self.board

        def print_board(self):
            """Print the minesweeper board for debugging"""
            printing_board = self.board.copy()
            printing_board = printing_board.astype(str)
            printing_board[printing_board == '0'] = '-'
            printing_board[printing_board == '-1'] = '*'
            print(printing_board)

        def neighbors(self, row, col):
            """Return a list of neighboring locations specified by row and col"""
            row_i = (0, -1, 1) if 0 < row < self.height - 1 else ((0, -1) if row > 0 else (0, 1))
            col_i = (0, -1, 1) if 0 < col < self.width - 1 else ((0, -1) if col > 0 else (0, 1))
            return starmap((lambda a, b: [row + a, col + b]), product(row_i, col_i))

    def __init__(self, difficulty, first_click):
        self.difficulty = difficulty
        self.height = difficulty['height']
        self.width = difficulty['width']
        self.num_mines = difficulty['num_mines']
        self.total_opens = self.height * self.width - self.num_mines
        self.solved_opens = 0

        self.board = self.MinesweeperBoard(self.height, self.width, self.num_mines, first_click)
        self.state = np.array([[HIDDEN] * self.width] * self.height, np.int32)
        self.display_board = np.array([[HIDDEN] * self.width] * self.height, np.int32)
        self.grids_3bv = []
        self.solved_3bv = 0
        self.total_3bv = self.get_total_3bv()
        self.is_finished = False
        self.result = False

        self.start_time = time.time()
        self.finish_time = -1

        #self.board.print_board()
        
    def get_result(self):
        return self.result

    def finished(self):
        return self.is_finished

    def get_time(self):
        return time.time() - self.start_time

    def get_finish_time(self):
        return self.finish_time

    def get_current_3bv(self):
        return self.solved_3bv

    def get_3bv(self):
        return self.total_3bv

    def get_board(self):
        return self.display_board

    def get_mines(self):
        return self.board.get_mines()

    def win(self):
        self.finish_time = self.get_time()
        for pos in range(self.height * self.width):
            row = pos // self.width
            col = pos % self.width
            if self.board.get_pos(row,col) == _MINE_:
                self.display_board[row][col] = FLAGGED
        self.is_finished = True
        self.result = True

    def lose(self, death_row, death_col):
        self.finish_time = self.get_time()
        self.is_finished = True
        self.result = False
        for pos in range(self.height * self.width):
            row = pos // self.width
            col = pos % self.width
            if self.state[row][col] == FLAGGED:
                if self.board.get_pos(row, col) != _MINE_:
                    self.display_board[row][col] = MISFLAGGED
                else:
                    self.display_board[row][col] = FLAGGED
            elif self.board.get_pos(row,col) == _MINE_:
                self.display_board[row][col] = _MINE_
        self.display_board[death_row][death_col] = DEATH_MINE
        
    def get_total_3bv(self):
        total_3bv = 0

        empty_grid_exist = True
        while empty_grid_exist:
            for pos in range(self.height*self.width):
                row = pos // self.width
                col = pos % self.width
                if self.board.get_pos(row, col) == _EMPTY_ and self.state[row][col] != OPEN:
                    self.cascade(row, col)
                    total_3bv += 1
                    break
            else:
                empty_grid_exist = False
        for pos in range(self.height*self.width):
            row = pos // self.width
            col = pos % self.width
            if self.board.get_pos(row, col) != _MINE_ and self.state[row][col] != OPEN:
                self.grids_3bv.append((row,col,))
                total_3bv += 1

        self.state = np.array([[HIDDEN] * self.width] * self.height, np.int32)
        self.display_board = np.array([[HIDDEN] * self.width] * self.height, np.int32)
        self.solved_opens = 0
        return total_3bv

    def neighbors(self, row, col):
        """Return a list of neighboring locations specified by row and col"""
        row_i = (0, -1, 1) if 0 < row < self.height - 1 else ((0, -1) if row > 0 else (0, 1))
        col_i = (0, -1, 1) if 0 < col < self.width - 1 else ((0, -1) if col > 0 else (0, 1))
        return starmap((lambda a, b: [row + a, col + b]), product(row_i, col_i))

    def flag(self, row, col):
        changed = []
        """Flag a coordinate. equiv to right click"""
        if self.state[row][col] == HIDDEN:
            self.state[row][col] = FLAGGED
            self.display_board[row][col] = FLAGGED
            changed.append((row, col,))
        elif self.state[row][col] == FLAGGED:
            self.state[row][col] = HIDDEN
            self.display_board[row][col] = HIDDEN
            changed.append((row, col,))
        return changed

    def open(self, row, col):
        self.state[row][col] = OPEN
        self.display_board[row][col] = self.board.get_pos(row, col)
        self.solved_opens += 1

    def click(self, row, col):
        """Click a position. Equiv to left click"""
        # Changed method name to click
        # Left click only opens grid if the grid is hidden
        # Left click does nothing when the grid is already open or flagged
        
        changed = []
        changed.append((row, col,))
        if self.state[row][col] == HIDDEN:
            if self.board.get_pos(row, col) == _MINE_:
                self.lose(row, col)
                return changed   # indicate a mine is clicked
            self.open(row, col)
            # cascade if clicked on EMPTY
            if self.board.get_pos(row, col) == _EMPTY_:
                self.solved_3bv += 1
                changed.extend(self.cascade(row, col))
            if (row, col,) in self.grids_3bv:
                self.solved_3bv += 1
            if self.solved_opens == self.total_opens:
                self.win()

        return changed     # nothing is wrong

    def cascade(self, row, col):
        """
            Opens all surrounding grids, recurse cascade if surrounding grid
            is also EMPTY
        """
        changed = []
        for nrow, ncol in self.neighbors(row, col):
            if self.state[nrow][ncol] == HIDDEN:
                self.open(nrow, ncol)
                changed.append((nrow, ncol,))
                if self.board.get_pos(nrow, ncol) == _EMPTY_:
                    changed.extend(self.cascade(nrow, ncol))
            if self.solved_opens == self.total_opens:
                self.win()

        return changed

    def chord(self, row, col):
        """
        click with both left and right to open a flagged region
        only work if there are as many flags at the neighbors of the
        operation as the number on that grid
        This operation may trigger mines if flags are not correctly
        on mines
        :param row: the row of this operation
        :param col: the column of this operation
        :return: True or False, true indicating no mine triggered
                                false indicating mines triggered
        """
        changed = []
        num_flags = 0

        for nrow, ncol in self.neighbors(row, col):
            if self.state[nrow][ncol] == FLAGGED:
                num_flags += 1

        if num_flags == self.board.get_pos(row, col):
            for nrow, ncol in self.neighbors(row, col):
                if self.display_board[nrow, ncol] == HIDDEN:
                    changed.extend(self.click(nrow, ncol))
      
        return changed

    def print_board(self):
        """Print the minesweeper board for displaying"""
        printing_board = self.display_board.copy()
        printing_board = printing_board.astype(str)
        printing_board[printing_board == '101'] = '-'
        printing_board[printing_board == '102'] = 'F'
        printing_board[printing_board == '103'] = 'X'
        printing_board[printing_board == '104'] = 'E'
        printing_board[printing_board == '-1'] = '*'
        print(printing_board)

    def get_pos(self, row, col):
        return self.display_board[row][col]
