from Minesweeper import *
from NNModel import *
from PyQt5.Qt import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import *
import time

class NNAgent(object):

    def __init__(self, mineGUI, speed_3bv):
        self.GUI = mineGUI
        self.net = self.loadModel()
        self.nRows = self.GUI.difficulty['height']
        self.nCols = self.GUI.difficulty['width']
        self.speed_3bv = speed_3bv
        self.delay = 1.0 / speed_3bv
        self.delay_milli = self.delay * 1000
        self.first_move = True

    def play(self):
        if self.first_move:
            init_row = int(self.nRows / 2)
            init_col = int(self.nCols / 2)

            mouseEvent1 = QMouseEvent(QEvent.MouseButtonPress, QPoint(init_row*16+8, init_col*16+8), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            mouseEvent2 = QMouseEvent(QEvent.MouseButtonRelease, QPoint(init_row*16+8, init_col*16+8), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            QCoreApplication.postEvent(self.GUI.board, mouseEvent1)
            QCoreApplication.postEvent(self.GUI.board, mouseEvent2)
            QCoreApplication.processEvents()

            self.first_move = False
        elif not self.GUI.finished:
            curr_inputs, curr_mask = getInputsFromGame(self.GUI.mgame)

            predict_input = curr_inputs.unsqueeze(0)
            predict_mask = curr_mask.unsqueeze(0)
            out = self.net(predict_input, predict_mask)

            # choose best cell
            selected = torch.argmin(out[0][0] + curr_inputs[0])
            selected_row = int(selected / self.nCols)
            selected_col = int(selected % self.nCols)

            mouseEvent1 = QMouseEvent(QEvent.MouseButtonPress, QPoint(selected_row * 16 + 8, selected_col * 16 + 8),
                                      Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            mouseEvent2 = QMouseEvent(QEvent.MouseButtonRelease, QPoint(selected_row * 16 + 8, selected_col * 16 + 8),
                                      Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
            QCoreApplication.postEvent(self.GUI.board, mouseEvent1)
            QCoreApplication.postEvent(self.GUI.board, mouseEvent2)
            QCoreApplication.processEvents()

    def loadModel(self):
        net = NNModel()

        if self.GUI.difficulty == DIFF_BEGINNER:
            checkpoint = torch.load('./trainedModels/testModel_beginner.pt')
        elif self.GUI.difficulty == DIFF_INTERMED:
            pass
        elif self.GUI.difficulty == DIFF_EXPERT:
            pass

        net.load_state_dict(checkpoint['model_state_dict'])
        return net
