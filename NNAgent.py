from Minesweeper import *
from NN import *
import numpy as np



class NNAgent(object):

    def __init__(self, mineGUI):
        self.NN = NN()
        self.win = 0
        self.lose = 0
        self.GUI = mineGUI
        self.mGame = None
        self.display_board = None
        self.board = None

    def play(self):
        self.mGame = Minesweeper(DIFF_BEGINNER, (0, 0))
        self.GUI.mgame = self.mGame

        self.GUI.board.on_click(0, 0)
        #self.mGame.click(0, 0)

        self.display_board = self.mGame.get_board()
        self.board = self.mGame.board
        while not self.mGame.is_finished:
            for i, j in self.perimeter_grids(self.display_board):
                prob = self.NN.predict(self.to_NNstate(i, j))
                if prob < 0.1:
                    self.GUI.board.on_flag(i, j)
                    #self.mGame.flag(i, j)

            prob_array = np.zeros((self.mGame.difficulty['width'], self.mGame.difficulty['height']))
            max_i = None
            max_j = None
            for i, j in self.perimeter_grids(self.display_board):
                possibility = self.NN.predict(self.to_NNstate(i, j))
                if possibility > prob_array.max():
                    max_i, max_j = i, j
                    prob_array[i, j] = possibility

            self.GUI.board.on_click(i, j)
            #self.mGame.click(max_i, max_j)
            #print(self.board.board)
            y = 0.0 if self.board.board[max_i, max_j] == -1 else 1.0
            self.NN.train(self.to_NNstate(max_i, max_j), [[y]])
            #self.mGame.print_board()

        if self.mGame.result:
            self.win += 1
        else:
            self.lose += 1

        print('W:{} - L:{}'.format(self.win, self.lose))

    def perimeter_grids(self, board):
        grids = []
        board_list = board.tolist()
        #print(board)
        for i in range(len(board_list)):
            for j in range(len(board_list[0])):
                if board[i, j] == HIDDEN:
                    grids.append((i, j))
        return grids

    def to_NNstate(self, i, j):

        state = np.array([])
        for row in range(5):
            for col in range(5):
                sample = np.zeros(12)
                grid_x = i - 2 + row
                grid_y = j - 2 + col

                if grid_x < 0  or grid_y < 0 or grid_x >= self.board.height or grid_y >= self.board.width:
                    sample[11] = 1.0
                elif self.display_board[grid_x, grid_y] == HIDDEN:
                    sample[9] = 1.0
                elif self.display_board[grid_x, grid_y] == FLAGGED:
                    sample[10] = 1.0
                elif self.display_board[grid_x, grid_y] >= 0 and self.display_board[grid_x, grid_y] <= 8:
                    sample[self.display_board[grid_x, grid_y]] = 1.0
                state = np.concatenate((state, sample))
        return state.reshape(1, 300)

if __name__ == '__main__':
    agent = NNAgent(None)
    for i in range(100):
        agent.play()
    agent.NN.save()
