import torch
from torch import nn, optim
import numpy as np
import copy
import os

class Agent(nn.Module):

    def __init__(self, num_actions: int, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.to(self.device)

        self._iter = 0
        self._episode_num = 1
        self._prev_state = None

        self.num_actions = num_actions
        self._init_network(self.num_actions)
        self.target_network_update_frequency = 10000
        self._target_network_update_counter = 0

        self.replay_buffer_size = 1000000
        self.save_replay_buffer = True
        self._init_replay_buffer(self.replay_buffer_size)
        self.train_start_time = 12500

        self.batch_size = 32
        self.lr = 0.00025
        self.momentum = 0.95
        self.min_squared_gradient = 0.01
        self.optim = optim.RMSprop(self.network.parameters(), lr=self.lr, momentum=self.momentum, eps=self.min_squared_gradient)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.start_eps = 1
        self.end_eps = 0.1
        self.eps_schedule_steps = 250000
        self._init_eps_schedule(self.start_eps, self.end_eps, self.eps_schedule_steps)
        self.eval_eps = 0.05

        self.gamma = 0.99

    def reset(self, action: int, reward: float) -> None:
        if self.training:
            if self._prev_state is not None:
                self._add_to_replay_buffer(self._prev_state, action, reward, None)
                self._prev_state = None
            self._episode_num += 1

    def has_checkpoint(self) -> None:
        return os.path.exists("agent.pt")

    def save_checkpoint(self) -> None:
        torch.save({
            "iter": self._iter,
            "episode_num": self._episode_num,
            "target_network_update_counter": self._target_network_update_counter,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
        }, "agent.pt")
        if self.save_replay_buffer:
            torch.save({
                "curr_replay_buffer_idx": self._curr_replay_buffer_idx,
                "replay_buffer": self.replay_buffer,
            }, "replay_buffer.pt")

    def load_checkpoint(self) -> tuple[int, int]:
        checkpoint = torch.load(
            "agent.pt",
            weights_only=True,
        )
        if self.training and os.path.exists("replay_buffer.pt"):
            replay_buffer_checkpoint = torch.load(
                "replay_buffer.pt",
                weights_only=True,
            )
            self.replay_buffer = replay_buffer_checkpoint["replay_buffer"]
            self._curr_replay_buffer_idx = replay_buffer_checkpoint["curr_replay_buffer_idx"]
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self._iter = checkpoint["iter"]
        self._episode_num = checkpoint["episode_num"]
        self._target_network_update_counter = checkpoint["target_network_update_counter"]
        self._update_current_eps()
        return self._iter, self._episode_num


    def _init_network(self, num_actions: int) -> None:
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        ).to(self.device)
        self.target_network = copy.deepcopy(self.network)

    def _init_replay_buffer(self, replay_buffer_size: int) -> None:
        self.replay_buffer = [None for _ in range(replay_buffer_size)]
        self._curr_replay_buffer_idx = 0
    
    def _add_to_replay_buffer(self, prev_state: torch.Tensor, action: int, reward: float, state: torch.Tensor) -> None:
        prev_state = (prev_state * 255).to(device=torch.device("cpu"), dtype=torch.uint8)
        action = action
        reward = reward
        state = None if state is None else (state * 255).to(device=torch.device("cpu"), dtype=torch.uint8)

        self.replay_buffer[self._curr_replay_buffer_idx] = (prev_state, action, reward, state)
        self._curr_replay_buffer_idx = (self._curr_replay_buffer_idx + 1) % self.replay_buffer_size

    def _sample_from_replay_buffer(self, batch_size: int) -> None:
        prev_states = []
        actions = []
        rewards = []
        nonterminal_states = []
        nonterminal_states_mask = []

        for batch_idx, sample_idx in enumerate(torch.randint(high=min(self._iter, self.replay_buffer_size), size=(batch_size, ))):
            prev_state, action, reward, state = self.replay_buffer[sample_idx]
            prev_states.append(prev_state)
            actions.append(action)
            rewards.append(reward)
            if state is not None:
                nonterminal_states.append(state)
                nonterminal_states_mask.append(batch_idx)
        prev_states = (torch.stack(prev_states)/255.0).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        nonterminal_states = (torch.stack(nonterminal_states)/255.0).to(self.device)
        nonterminal_states_mask = torch.tensor(nonterminal_states_mask).to(self.device)
        batch = (prev_states, actions, rewards, nonterminal_states, nonterminal_states_mask)
        return batch
    
    def get_curr_eps(self) -> torch.Tensor:
        return self._curr_eps if self.training else torch.tensor(self.eval_eps)

    def _init_eps_schedule(self, start_eps: float, end_eps: float, eps_schedule_steps: int) -> None:
        self._eps_schedule = torch.linspace(start=start_eps, end=end_eps, steps=eps_schedule_steps)
        self._update_current_eps()

    def _update_current_eps(self) -> None:
        if self._iter < self.eps_schedule_steps:
            self._curr_eps = self._eps_schedule[self._iter]
        else:
            self._curr_eps = self._eps_schedule[-1]

    def train_step(self) -> None:
        if self._iter < self.train_start_time:
            return

        batch = self._sample_from_replay_buffer(self.batch_size)
        prev_states, actions, rewards, nonterminal_states, nonterminal_states_mask = batch

        qs = torch.zeros((self.batch_size, )).to(self.device)
        qs[nonterminal_states_mask] = torch.max(self.target_network(nonterminal_states), dim=-1).values
        targets = rewards + self.gamma*qs

        prev_qs = self.network(prev_states)[torch.arange(self.batch_size), actions]

        self.optim.zero_grad()
        loss = self.loss_fn(prev_qs, targets)
        loss.backward()
        self.optim.step()

        if self._target_network_update_counter < self.target_network_update_frequency:
            self._target_network_update_counter += 1
        else:
            self.target_network = copy.deepcopy(self.network)
            self._target_network_update_counter = 0

    def forward(self, state: torch.Tensor, action: int, reward: float) -> int | tuple[int, torch.Tensor]:

        if self.training:
            if self._prev_state is not None:
                self._add_to_replay_buffer(self._prev_state, action, reward, state)
            self._prev_state = state
        
        if self._iter < self.train_start_time or torch.bernoulli(self.get_curr_eps()) == 1:
            action = torch.randint(high=self.num_actions, size=(1, )).item()
        else:
            qs = self.network(state.unsqueeze(0))
            action = torch.argmax(qs, -1).squeeze(0).item()

        if self.training:
            self._iter += 1
            self._update_current_eps()
            
        return action