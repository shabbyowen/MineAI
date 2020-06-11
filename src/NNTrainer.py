import matplotlib.pyplot as plt
from sklearn.utils import shuffle, resample
import torch
import torch.nn as nn
import torch.optim as optim

from Minesweeper import *
from Minesweeper import _MINE_
from Minesweeper import Minesweeper as mGame

from NNModel import *

# If there are GPUs, choose the first one for computing. Otherwise use CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# If 'cuda:0' is printed, it means GPU is available.

net = NNModel()
loss_func = nn.BCELoss()
opt = optim.Adam(net.parameters())

def fit(x, masks, y, epochs):
  avg_losses = []  # Avg. losses.
  print_freq = 10  # Print frequency.

  inputs = x
  labels = y

  for epoch in range(epochs):  # Loop over the dataset multiple times.
    running_loss = 0.0  # Initialize running loss.

    # Move the inputs to the specified device.
    #inputs, masks, labels = shuffle(inputs, masks, labels, replace=True)
    #inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

    # Zero the parameter gradients.
    opt.zero_grad()

    # Forward step.
    outputs = net(inputs, masks)
    loss = loss_func(outputs, labels.detach())

    # Backward step.
    loss.backward()

    # Optimization step (update the parameters).
    opt.step()

    # Print statistics.
    avg_losses.append(loss.item())

  return float(torch.mean(torch.tensor(avg_losses)))

def trainMineAI(nBatches, nSamples, nEpochsPerBatch, difficulty, train_new=False, model_file=None):
  """
  Args:
      nBatches: number of batches to train
      nSamples: number of games per batch -> fit(batch_size)
      nEpochsPerBatch: training epochs per batch
      nRows: minesweeper game board length  (# of rows)
      nCols: minesweeper game board width   (# of cols)
  """
  batch_losses = []
  mean_3bv_solves = []
  batch_num_already_trained = 0
  if not train_new:
    checkpoint = torch.load(model_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    batch_num_already_trained = checkpoint['batch_trained']
    batch_losses = checkpoint['batch_losses']
    mean_3bv_solves = checkpoint['3bv_solves']

  nRows = difficulty['height']
  nCols = difficulty['width']

  x = torch.zeros((nSamples, 11, nRows, nCols))
  masks = torch.zeros((nSamples, 1, nRows, nCols))
  y = torch.zeros((nSamples, 1, nRows, nCols))


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
        curr_inputs, curr_mask = getInputsFromGame(game)
        x[samplesTaken] = curr_inputs
        masks[samplesTaken] = curr_mask

        # make probability predictions
        # print(curr_inputs.shape)
        # print(mask.shape)
        # print(curr_inputs.unsqueeze(0).shape)
        # print(mask.unsqueeze(0).shape)
        predict_input = curr_inputs.unsqueeze(0)
        predict_mask = curr_mask.unsqueeze(0)
        out = net(predict_input, predict_mask)

        # choose best remaining cell
        selected = torch.argmin(out[0][0] + curr_inputs[0])
        selected_row = int(selected / nCols)
        selected_col = int(selected % nCols)
        game.click(selected_row, selected_col)

        # find truth
        truth = out
        truth[0, 0, selected_row, selected_col] = 1.0 if game.board.board[selected_row][selected_col] == _MINE_ else 0.0
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
    mean_3bv_solves.append(mean3BVSolved)
    batch_num = batch_num_already_trained + i + 1
    print('Games played in batch {}: {} '.format(batch_num, gamesPlayed))
    print('Mean 3BV solved percent in batch {}: {}%'.format(batch_num, mean3BVSolved * 100))
    print('Proportion of games won in batch {}: {}%'.format(batch_num, propGamesWon * 100))

    # train
    batch_loss = fit(x, masks, y, nEpochsPerBatch)
    batch_losses.append(batch_loss)
    print('Finished batch number {}/{} Training in current session. Batch loss: {}'.format(i+1, nBatches, batch_loss))

    # save model every 10 batch
    if (i + 1) % 5 == 0:
      torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'batch_losses': batch_losses,
        '3bv_solves': mean_3bv_solves,
        'batch_trained': batch_num
      }, model_file)

  # plt.plot(batch_losses)
  # plt.xlabel('batch index')
  # plt.ylabel('batch loss')
  # plt.show()

def testMineAI(nGames, difficulty, model_file):
  # Load model from model_file
  checkpoint = torch.load(model_file)
  net.load_state_dict(checkpoint['model_state_dict'])

  nRows = difficulty['height']
  nCols = difficulty['width']
  solved_3bv = 0
  gamesWon = 0
  print_freq = 100
  for i in range(nGames):
    if (i % print_freq) == 0:
      print("Playing game " + str(i + 1) + "...")

    # initiate game, choose middle as first click
    game = mGame(difficulty, (int(nRows / 2), int(nCols / 2)))
    game.click(int(nRows / 2), int(nCols / 2))

    while not game.is_finished:

      curr_inputs, curr_mask = getInputsFromGame(game)

      predict_input = curr_inputs.unsqueeze(0)
      predict_mask = curr_mask.unsqueeze(0)
      out = net(predict_input, predict_mask)

      # choose best cell
      selected = torch.argmin(out[0][0] + curr_inputs[0])
      selected_row = int(selected / nCols)
      selected_col = int(selected % nCols)
      game.click(selected_row, selected_col)

    solved_3bv += game.get_current_3bv() / game.get_3bv()
    if game.result:
      gamesWon += 1
  mean3BVSolved = float(solved_3bv) / nGames
  propGamesWon = float(gamesWon) / nGames
  print('Mean 3BV solved percent in game {}: {}%'.format(i, mean3BVSolved * 100))
  print('Proportion of games won in game {}: {}%'.format(i, propGamesWon * 100))

# Train Beginner level network
#trainMineAI(nBatches=100, nSamples=1000, nEpochsPerBatch=10, difficulty=DIFF_BEGINNER, train_new=True, model_file='./trainedModels/testModel.pt')
#trainMineAI(nBatches=50, nSamples=1000, nEpochsPerBatch=10, difficulty=DIFF_BEGINNER, train_new=False, model_file='./trainedModels/testModel.pt')

# Train InterMediate level network
#trainMineAI(nBatches=50, nSamples=1000, nEpochsPerBatch=10, difficulty=DIFF_INTERMED, train_new=True, model_file='./trainedModels/testModel.pt')
#trainMineAI(nBatches=50, nSamples=1000, nEpochsPerBatch=10, difficulty=DIFF_INTERMED, train_new=False, model_file='./trainedModels/testModel.pt')
#trainMineAI(nBatches=50, nSamples=1000, nEpochsPerBatch=10, difficulty=DIFF_INTERMED, train_new=False, model_file='./trainedModels/testModel.pt')

# Train Expert level network
#trainMineAI(nBatches=50, nSamples=1000, nEpochsPerBatch=1, difficulty=DIFF_EXPERT, train_new=True, model_file='./trainedModels/testModel.pt')
trainMineAI(nBatches=50, nSamples=1000, nEpochsPerBatch=1, difficulty=DIFF_EXPERT, train_new=False, model_file='./trainedModels/testModel.pt')

# Test model
#testMineAI(10000, DIFF_BEGINNER, './trainedModels/testModel_beginner.pt')
#testMineAI(1000, DIFF_BEGINNER, './trainedModels/testModel_intermed.pt')
#testMineAI(1000, DIFF_INTERMED, './trainedModels/testModel_intermed.pt')
#testMineAI(1000, DIFF_BEGINNER, './trainedModels/testModel_expert.pt')
#testMineAI(1000, DIFF_INTERMED, './trainedModels/testModel_expert.pt')
#testMineAI(1000, DIFF_EXPERT, './trainedModels/testModel_expert.pt')
#testMineAI(1000, DIFF_BEGINNER, './trainedModels/testModel.pt')
#testMineAI(1000, DIFF_INTERMED, './trainedModels/testModel.pt')
