import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Определение класса нейронной сети для игры в "крестики-нолики"
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Создание экземпляра нейронной сети
net = TicTacToeNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)  # Перемещение нейронной сети на GPU (если доступно)

optimizer = optim.SGD(net.parameters(), lr=0.1)
criterion = nn.MSELoss()  # Потери среднеквадратичной ошибки для оценки качества ходов

# Определение функции, которая генерирует состояние игрового поля
def generate_game_state():
    # Создайте пустое игровое поле 3x3 (0 - пусто, 1 - крестик, -1 - нолик)
    game_state = np.zeros((3, 3), dtype=np.float32)
    return game_state

# Определение функции для выбора хода на основе оценок нейронной сети
def select_move(game_state, net):
    game_state_flat = game_state.flatten()
    game_state_tensor = torch.tensor(game_state_flat, dtype=torch.float32)
    game_state_tensor = game_state_tensor.to(device)  # Перемещение данных на GPU
    action_scores = net(game_state_tensor)
    valid_moves = (game_state_flat == 0)  # Определение доступных ходов
    action_scores[~valid_moves] = -1e9  # Заглушка для недопустимых ходов
    best_action = torch.argmax(action_scores)
    return best_action.item()

# Определение функции для оценки победителя игры
def evaluate_winner(game_state):
    # Проверяем все возможные способы выигрыша
    for row in game_state:
        if np.all(row == 1) or np.all(row == -1):
            return row[0]

    for col in game_state.T:
        if np.all(col == 1) or np.all(col == -1):
            return col[0]

    if np.all(np.diag(game_state) == 1) or np.all(np.diag(game_state) == -1):
        return game_state[0, 0]

    if np.all(np.diag(np.fliplr(game_state)) == 1) or np.all(np.diag(np.fliplr(game_state)) == -1):
        return game_state[0, 2]

    # Если нет победителя и не осталось пустых ячеек, то это ничья
    if not np.any(game_state == 0):
        return 0

    # Если игра продолжается, возвращаем None
    return None

# Основной цикл обучения
# Основной цикл обучения
epochs = 10000
for epoch in range(epochs):
    round_number = epoch + 1  # Добавляем номер раунда

    while True:
        game_state = generate_game_state()  # Создаем новое состояние для каждого раунда
        winner = None  # Перемещаем инициализацию winner в начало игры

        while winner is None:  # Играем, пока нет победителя
            # Выбор хода с использованием нейронной сети
            action = select_move(game_state, net)

            # Реализация хода игрока (случайный ход)
            valid_moves = np.argwhere(game_state == 0)
            if len(valid_moves) > 0:
                random_choice = random.choice(valid_moves)
                game_state[random_choice] = 1

            # Оценка победителя или ничьей и вычисление целевой оценки
            winner = evaluate_winner(game_state)

            if winner is None:
                # Перемещение данных на GPU
                game_state_flat = game_state.flatten()
                game_state_tensor = torch.tensor(game_state_flat, dtype=torch.float32).to(device)
                target_score = torch.zeros(9, dtype=torch.float32).to(device)
                target_score[action] = 1.0

                # Обновление нейронной сети на основе полученной целевой оценки
                optimizer.zero_grad()
                action_scores = net(game_state_tensor)
                loss = criterion(action_scores, target_score)
                loss.backward()
                optimizer.step()

        # Вывод результата раунда
        print(f"Round {round_number}: Winner = {winner}")

        # Условие окончания игры: победитель определен или нет доступных ходов (ничья)
        if winner is not None or len(valid_moves) == 0:
            break

# Сохранение весов нейронной сети
torch.save(net.state_dict(), 'tic_tac_toe_net.pth')





