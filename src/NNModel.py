import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F

from Minesweeper import *

def getInputsFromGame(mGame):
  state = torch.tensor(mGame.state)
  display_board = torch.tensor(mGame.display_board)

  inputs = torch.zeros((11, mGame.height, mGame.width))

  # channel 0: binary revealed map
  inputs[0] = torch.where(state == HIDDEN, torch.tensor(0.0), torch.tensor(1.0))
  mask = torch.where(state == HIDDEN, torch.tensor(1.0), torch.tensor(0.0))

  # channel 1: for zero padding detecting game board edge
  inputs[1] = torch.ones((mGame.height, mGame.width))

  # channel 2-10: numeric one-hot encoding
  for i in range(9):
    inputs[i + 2] = torch.where(display_board == i, torch.tensor(1.0), torch.tensor(0.0))

  return inputs, mask


class NNModel(nn.Module):
  def __init__(self):
    # Initialization.
    super(NNModel, self).__init__()
    self.conv1 = nn.Conv2d(11, 64, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.conv5 = nn.Conv2d(64, 1, 1, stride=1, padding=0)
    self.sigmoid = nn.Sigmoid()

    torch_init.xavier_normal_(self.conv1.weight)
    torch_init.xavier_normal_(self.conv2.weight)
    torch_init.xavier_normal_(self.conv3.weight)
    torch_init.xavier_normal_(self.conv4.weight)
    torch_init.xavier_normal_(self.conv5.weight)

  def forward(self, x, revealed):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.sigmoid(self.conv5(x))
    x = x * revealed
    return x