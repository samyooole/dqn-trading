import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from utils import Portfolio
import numpy as np

class DQNTrader(Portfolio):
    def __init__(self, state_dim, balance, is_eval=False):
        super().__init__(balance=balance)

        self.model_type = 'DQN'
        self.state_dim = state_dim
        self.action_dim = 3
        self.memory = deque(maxlen=100)
        self.buffer_size = 60
        self.gamma = 0.95
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.01  # minimum exploration rate
        self.epsilon_decay = 0.995  # decrease exploration rate over time
        self.is_eval = is_eval

        # Instantiate model
        self.model = self.model_instantiate()

    def model_instantiate(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )
        return model

    def reset(self):
        self.reset_portfolio()
        self.epsilon = 1.0

    def remember(self, state, actions, reward, next_state, done):
        self.memory.append((state, actions, reward, next_state, done))

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state)
            options = self.model(state_tensor)
            return torch.argmax(options.detach()).item()

    def experience_replay(self):
        mini_batch = random.sample(self.memory, self.buffer_size)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for state, actions, reward, next_state, done in mini_batch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)
            actions_tensor = actions.clone().detach()

            if not done:
                Q_target_value = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()
            else:
                Q_target_value = reward

            next_actions = self.model(state_tensor)
            next_actions_detach = next_actions.detach().clone()  # Detach and clone the tensor
            next_actions_detach[0][torch.argmax(actions_tensor)] = Q_target_value

            # Compute loss
            target_actions = self.model(state_tensor)
            loss = loss_fn(next_actions_detach, target_actions)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()
