import torch
import torch.nn as nn
import numpy as np
import os


class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_model(path):
    model = TicTacToeNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def print_board(board):
    symbols = {0: "   ", 1: " X ", -1: " O "}
    print("Current board:")
    for row in board:
        print("   |   ".join(symbols[x] for x in row))
        print("       " + "-" * 19)


def get_player_move(board):
    move = input("Enter your move (row,col), where top-left is (0,0): ")
    row, col = map(int, move.split(","))
    while board[row][col] != 0:
        print("Invalid move. Cell is already occupied.")
        move = input("Enter your move (row,col), where top-left is (0,0): ")
        row, col = map(int, move.split(","))
    return row, col


def select_move(net, board):
    board_flat = board.flatten()
    board_tensor = torch.tensor(board_flat, dtype=torch.float32)
    with torch.no_grad():
        action_scores = net(board_tensor)
    valid_moves = board_flat == 0
    action_scores[~valid_moves] = -1e9
    best_action = torch.argmax(action_scores)
    move = np.unravel_index(best_action.item(), board.shape)
    print(
        f"Neural network ({'X' if player == 1 else 'O'}) makes a move at ({move[0]},{move[1]})"
    )
    return move


def evaluate_winner(board):
    for row in board:
        if np.all(row == 1):
            return 1
        if np.all(row == -1):
            return -1
    for col in board.T:
        if np.all(col == 1):
            return 1
        if np.all(col == -1):
            return -1
    if np.all(np.diag(board) == 1) or np.all(np.diag(board) == -1):
        return board[0, 0]
    if np.all(np.diag(np.fliplr(board)) == 1) or np.all(
        np.diag(np.fliplr(board)) == -1
    ):
        return board[0, 2]
    if not np.any(board == 0):
        return 0
    return None


# Asking for model
version = input("Enter the model version (e.g., v1, v2): ")
model_path = os.path.join("models", f"tic_tac_toe_net_{version}.pth")
net = load_model(model_path)

# Asking to chose a side
player_choice = int(input("Choose your side: Enter 1 for X or -1 for O: "))


board = np.zeros((3, 3), dtype=int)

if player_choice == 1:
    print("You are playing as X.")
    print("Neural network is playing as O.")
    player = 1  # Playing as X
else:
    print("You are playing as O.")
    print("Neural network is playing as X.")
    player = -1  # Playing as 0

while True:
    print_board(board)
    if player == player_choice:
        print(f"Your turn ({'X' if player_choice == 1 else 'O'}).")
        row, col = get_player_move(board)
    else:
        print(f"Neural network's turn ({'X' if player == -1 else 'O'}).")
        row, col = select_move(net, board * player)
    board[row, col] = player

    winner = evaluate_winner(board)
    if winner is not None or np.all(board != 0):
        print_board(board)
        if winner == player_choice:
            print("Congratulations! You won!")
        elif winner != 0:
            print("Neural network won. Better luck next time!")
        else:
            print("It's a draw!")
        break

    player *= -1  # Changing sides
