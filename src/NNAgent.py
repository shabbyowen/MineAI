from Minesweeper import *
from NNModel import *
from PyQt5.Qt import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import *

from copy import deepcopy
import time, threading


class NNAgent(object):

    def __init__(self, mineGUI, model_file):
        self.GUI = mineGUI
        self.events = []
        self.net = NNModel()
        checkpoint = torch.load(model_file)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.delay = 1

    def append_click_event(self, row, col):
        mouseEvent1 = QMouseEvent(QEvent.MouseButtonPress, QPoint(col * 16 + 8, row * 16 + 8), Qt.LeftButton,
                                   Qt.LeftButton, Qt.NoModifier)
        mouseEvent2 = QMouseEvent(QEvent.MouseButtonRelease, QPoint(col * 16 + 8, row * 16 + 8),
                                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        self.events.append(mouseEvent1)
        self.events.append(mouseEvent2)

    def click_cell(self, row, col):
        mouseEvent1 = QMouseEvent(QEvent.MouseButtonPress, QPoint(col * 16 + 8, row * 16 + 8), Qt.LeftButton,
                                   Qt.LeftButton, Qt.NoModifier)
        mouseEvent2 = QMouseEvent(QEvent.MouseButtonRelease, QPoint(col * 16 + 8, row * 16 + 8),
                                  Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        QCoreApplication.postEvent(self.GUI.board, mouseEvent1)
        QCoreApplication.postEvent(self.GUI.board, mouseEvent2)
        QCoreApplication.processEvents()

    def play_in_background(self):
        nRows = self.GUI.difficulty['height']
        nCols = self.GUI.difficulty['width']
        init_row = int(nRows / 2)
        init_col = int(nCols / 2)

        self.click_cell(init_row, init_col)

        game = deepcopy(self.GUI.mgame)

        while not game.is_finished:
            curr_inputs, curr_mask = getInputsFromGame(game)

            predict_input = curr_inputs.unsqueeze(0)
            predict_mask = curr_mask.unsqueeze(0)
            out = self.net(predict_input, predict_mask)

            # choose best cell
            selected = torch.argmin(out[0][0] + curr_inputs[0])
            selected_row = int(selected / nCols)
            selected_col = int(selected % nCols)
            game.click(selected_row, selected_col)
            self.append_click_event(selected_row, selected_col)

    def post_events(self):
        if self.events:
            QCoreApplication.postEvent(self.GUI.board, self.events.pop(0))
            QCoreApplication.postEvent(self.GUI.board, self.events.pop(0))
            threading.Timer(self.delay, self.post_events).start()

    def play(self, speed_3bv):
        self.delay = 1 / speed_3bv
        self.play_in_background()
        self.post_events()


