import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time

epsilon = 0.1  # Вероятность случайного выбора хода


class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def print_board(board):
    symbols = {0: " ", 1: "X", -1: "O"}
    print("Current board:")
    for row in board:
        print(" | ".join(symbols[x] for x in row))
        print("-" * 5)


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


def update_network(net, optimizer, criterion, game_history, winner, device):
    for state, action in game_history:
        game_state_tensor = torch.tensor(state.flatten(), dtype=torch.float32).to(
            device
        )
        target_score = torch.zeros(9, dtype=torch.float32).to(device)
        if winner == 1:
            target_score[action] = 1.0
        elif winner == 0:
            target_score[action] = 0.5
        else:
            target_score[action] = 0
        optimizer.zero_grad()
        action_scores = net(game_state_tensor)
        loss = criterion(action_scores, target_score)
        loss.backward()
        optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Загрузка текущей модели (версии 1)
current_net = TicTacToeNet().to(device)
current_net.load_state_dict(torch.load("tic_tac_toe_net_v2.pth"))
current_net.eval()

# Создание новой модели (версии 2) для обучения
new_net = TicTacToeNet().to(device)

optimizer = optim.SGD(new_net.parameters(), lr=0.1)
criterion = nn.MSELoss()

epochs = 30000
for epoch in range(epochs):
    game_state = np.zeros((3, 3), dtype=np.float32)
    winner = None
    player = 1  # Start with new net (version 2)
    game_history = []

    while winner is None:

        net_to_use = new_net if player == 1 else current_net
        game_state_flat = game_state.flatten()
        game_state_tensor = torch.tensor(game_state_flat * player, dtype=torch.float32).to(device)

        
        if random.random() < epsilon:
            valid_moves = np.argwhere(game_state == 0)  # Получаем индексы пустых клеток
            random_move = random.choice(valid_moves)  # Выбираем случайный ход
            action_idx = tuple(random_move)  # Преобразуем в кортеж для индексации
        else:
            with torch.no_grad():
                action_scores = net_to_use(game_state_tensor)
            valid_moves = game_state_flat == 0
            action_scores[~valid_moves] = -1e9
            best_action = torch.argmax(action_scores)
            action_idx = np.unravel_index(best_action.item(), game_state.shape)



        game_state[action_idx] = player
        #print(f"Player {player} ({'new_net' if player == 1 else 'current_net'}): Move at {action_idx}")
        #print_board(game_state)
        
        

        if player == 1:  # Сохраняем историю только для новой сети
            game_history.append((game_state.copy(), np.ravel_multi_index(action_idx, game_state.shape)))

        winner = evaluate_winner(game_state)
        player *= -1  # Switch player

    update_network(new_net, optimizer, criterion, game_history, winner, device)

    # Optional: Log results for monitoring
    if epoch % 1 == 0:
        print(
            f"Epoch {epoch + 1}: Winner is {'new net' if winner == 1 else 'current net' if winner == -1 else 'none'}"
        )

# Save both models
torch.save(current_net.state_dict(), "tic_tac_toe_net_v1.pth")
torch.save(new_net.state_dict(), "tic_tac_toe_net_v2.pth")
