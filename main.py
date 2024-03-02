import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = TicTacToeNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.MSELoss()


def generate_game_state():
    game_state = np.zeros((3, 3), dtype=np.float32)
    return game_state


def select_move(game_state, net):
    game_state_flat = game_state.flatten()
    game_state_tensor = torch.tensor(game_state_flat, dtype=torch.float32)
    game_state_tensor = game_state_tensor.to(device)
    action_scores = net(game_state_tensor)
    valid_moves = game_state_flat == 0
    action_scores[~valid_moves] = -1e9
    best_action = torch.argmax(action_scores)
    return best_action.item()


def evaluate_winner(game_state):
    for row in game_state:
        if np.all(row == 1):
            return 1
        if np.all(row == -1):
            return -1

    for col in game_state.T:
        if np.all(col == 1):
            return 1
        if np.all(col == -1):
            return -1

    if np.all(np.diag(game_state) == 1) or np.all(np.diag(game_state) == -1):
        return game_state[0, 0]

    if np.all(np.diag(np.fliplr(game_state)) == 1) or np.all(
        np.diag(np.fliplr(game_state)) == -1
    ):
        return game_state[0, 2]

    if not np.any(game_state == 0):
        return 0

    return None


epochs = 30000
for epoch in range(epochs):
    game_state = generate_game_state()
    winner = None
    player = 1  # Start with player 1 (network)
    game_history = []

    while winner is None:
        action = select_move(game_state * player, net)
        action_idx = np.unravel_index(action, game_state.shape)
        game_state[action_idx] = player
        game_history.append((game_state.copy(), action))

        winner = evaluate_winner(game_state)
        if winner is not None:
            break

        # Switch player
        player *= -1

        # Random move for the opponent
        valid_moves = np.argwhere(game_state == 0)
        if len(valid_moves) > 0:
            random_choice = random.choice(valid_moves)
            game_state[tuple(random_choice)] = player
            player *= -1  # Switch back to the original player

        winner = evaluate_winner(game_state)

    # Update network after the game is complete
    for state, action in game_history:
        game_state_tensor = torch.tensor(
            state.flatten() * player, dtype=torch.float32
        ).to(device)
        target_score = torch.zeros(9, dtype=torch.float32).to(device)
        if winner == player:  # If the current player won
            target_score[action] = 1.0
        elif winner == 0:  # Draw
            target_score[action] = 0.5
        else:  # Loss
            target_score[action] = 0

        optimizer.zero_grad()
        action_scores = net(game_state_tensor)
        loss = criterion(action_scores, target_score)
        loss.backward()
        optimizer.step()

    if winner == 1:
        print(f"Epoch {epoch + 1}/{epochs}: Winner = Neural Network (Player 1)")
    elif winner == -1:
        print(f"Epoch {epoch + 1}/{epochs}: Winner = Random Opponent (Player 2)")
    else:
        print(f"Epoch {epoch + 1}/{epochs}: Draw")

torch.save(net.state_dict(), "tic_tac_toe_net.pth")
