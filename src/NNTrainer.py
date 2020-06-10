import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import torch.optim as optim

from Minesweeper import *
from Minesweeper import _MINE_
from Minesweeper import Minesweeper as mGame

# If there are GPUs, choose the first one for computing. Otherwise use CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# If 'cuda:0' is printed, it means GPU is available.

class Net(nn.Module):
  def __init__(self):
    # Initialization.
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(11, 64, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.conv5 = nn.Conv2d(64, 1, 1, stride=1, padding=0)

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
    x = torch.sigmoid(self.conv5(x))
    # print('------forward print()------')
    # print(x.shape)
    # print(revealed.shape)
    x = x * revealed
    return x


net = Net()  # Create the network instance.
#net.to(device)  # Move the network parameters to the specified device.

# We use cross-entropy as loss function.
loss_func = nn.BCELoss()
# We use stochastic gradient descent (SGD) as optimizer.
#opt = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
opt = optim.Adam(net.parameters(), lr=0.00001)


class MineSweeperDataset(torch.utils.data.Dataset):
  """Minesweeper dataset."""

  def __init__(self, inputs, labels):
    """
    Args:
        inputs (n, 11, dim1, dim2): input minesweeper channels
        masks  (n, 1,  dim1, dim2): binary revealed masks
        labels (n, 1,  dim1, dim2): binary mine map
    """
    self.inputs = inputs
    self.masks = inputs[:, [0], :, :]
    self.masks = torch.where(self.masks == 0, torch.tensor(1), torch.tensor(0))
    self.labels = labels
    print(self.inputs[0][0])
    print(self.masks[0][0])
    print(self.inputs[1][0])
    print(self.masks[1][0])

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return (self.inputs[idx], self.masks[idx], self.labels[idx])

def getInputsFromGame(mGame):
  state = torch.tensor(mGame.state)
  display_board = torch.tensor(mGame.display_board)

  inputs = torch.zeros((11, mGame.height, mGame.width))

  # channel 0: binary revealed map
  inputs[0] = torch.where(state == HIDDEN, torch.tensor(0), torch.tensor(1))

  # channel 1: for zero padding detecting game board edge
  inputs[1] = torch.ones((mGame.height, mGame.width))

  # channel 2-10: numeric one-hot encoding
  for i in range(9):
    inputs[i + 2] = torch.where(display_board == i, torch.tensor(1), torch.tensor(0))

  inputs_np = np.zeros((11, mGame.height, mGame.width))

  # channel 0: binary revealed map
  inputs_np[0] = np.where(mGame.state == HIDDEN, 0, 1)

  # channel 1: for zero padding detecting game board edge
  inputs_np[1] = np.ones((mGame.height, mGame.width))

  # channel 2-10: numeric one-hot encoding
  for i in range(9):
    inputs_np[i + 2] = np.where(mGame.display_board == i, 1, 0)

  return inputs

def fit(x, y, batch_size, epochs):
  avg_losses = []  # Avg. losses.
  print_freq = 10  # Print frequency.

  masks = x[:, [0], :, :]
  masks = torch.where(masks == 0, torch.tensor(1), torch.tensor(0))

  for epoch in range(epochs):  # Loop over the dataset multiple times.
    running_loss = 0.0  # Initialize running loss.

    # Move the inputs to the specified device.
    inputs, labels = x, y
    #inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

    # Zero the parameter gradients.
    opt.zero_grad()

    # print('------fit print------')
    # print(inputs.shape)
    # print(masks.shape)
    # print(labels.shape)

    # Forward step.
    outputs = net(inputs, masks)
    print(labels.device)
    loss = loss_func(outputs, labels.detach())

    # Backward step.
    loss.backward()

    # Optimization step (update the parameters).
    opt.step()

    # Print statistics.
    avg_losses.append(loss.item())

  return float(torch.mean(torch.tensor(avg_losses)))

def trainMineAI(nBatches, nSamples, nEpochsPerBatch, difficulty):
  """
  Args:
      nBatches: number of batches to train
      nSamples: number of games per batch -> fit(batch_size)
      nEpochsPerBatch: training epochs per batch
      nRows: minesweeper game board length  (# of rows)
      nCols: minesweeper game board width   (# of cols)
  """
  nRows = difficulty['height']
  nCols = difficulty['width']

  x = torch.zeros(
    (nSamples, 11, nRows, nCols))  # 11 channels: 1 for if has been revealed, 1 for is-on-board, 1 for each number
  masks = torch.zeros((nSamples, 1, nRows, nCols))
  y = torch.zeros((nSamples, 1, nRows, nCols))

  batch_losses = []

  print_freq = 100  # Print frequency.

  for i in range(nBatches):
    solved_3bv = 0
    gamesPlayed = 0
    gamesWon = 0
    samplesTaken = 0
    while samplesTaken < nSamples:

      # initiate game, first click in center
      game = mGame(difficulty, (int(nRows / 2), int(nCols / 2)))
      game.click(int(nRows / 2), int(nCols / 2))

      while not (game.is_finished or samplesTaken == nSamples):
        # get data input from current game board
        curr_inputs = getInputsFromGame(game)
        x[samplesTaken] = curr_inputs
        mask = torch.where(x[samplesTaken][0] == 0, torch.tensor(1), torch.tensor(0))

        # make probability predictions
        # print(curr_inputs.shape)
        # print(mask.shape)
        # print(curr_inputs.unsqueeze(0).shape)
        # print(mask.unsqueeze(0).shape)
        predict_input = curr_inputs.unsqueeze(0)
        predict_mask = mask.unsqueeze(0).unsqueeze(0)
        #predict_input = curr_inputs.unsqueeze(0).to(device)
        #predict_mask = mask.unsqueeze(0).unsqueeze(0).to(device)
        out = net(predict_input, predict_mask)
        #out = out.cpu()

        # choose best remaining cell
        selected = torch.argmin(
          out[0][0] + curr_inputs[0])  # add Xnow[0] so that already selected cells aren't chosen
        selected_row = int(selected / nCols)
        selected_col = selected % nCols
        game.click(selected_row, selected_col)

        # find truth
        truth = out
        truth[0, 0, selected_row, selected_col] = 1 if game.display_board[selected_row][selected_col] == _MINE_ else 0
        y[samplesTaken] = truth[0]

        if samplesTaken % print_freq == print_freq - 1:  # Print every several mini-batches.
          print('Samples taken: {} / {}'.format(samplesTaken, nSamples))
        samplesTaken += 1

      if game.is_finished:
        gamesPlayed += 1
        solved_3bv += game.get_current_3bv() / game.get_3bv()
        if game.result:
          gamesWon += 1

    if gamesPlayed > 0:
      mean3BVSolved = float(solved_3bv) / gamesPlayed
      propGamesWon = float(gamesWon) / gamesPlayed
    print('Games played in batch {}: {} '.format(i, gamesPlayed))
    print('Mean 3BV solved percent in batch {}: {}%'.format(i, mean3BVSolved * 100))
    print('Proportion of games won in batch {}: {}%'.format(i, propGamesWon * 100))

    # train
    batch_loss = fit(x, y, nSamples, nEpochsPerBatch)
    batch_losses.append(batch_loss)
    print('Finished batch number {}/{} Training. Batch loss: {}'.format(i, nBatches, batch_loss))

    # save model every 100 batch
    if (i + 1) % 10 == 0:
      torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'batch_losses': batch_losses
      }, "./trainedModels/testModel.pt")

  plt.plot(batch_losses)
  plt.xlabel('batch index')
  plt.ylabel('batch loss')
  plt.show()

trainMineAI(nBatches=100, nSamples=1000, nEpochsPerBatch=1, difficulty=DIFF_BEGINNER)