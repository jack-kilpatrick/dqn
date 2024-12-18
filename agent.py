import torch
from torch import nn, optim
import copy

# Approx. replay memory size for 1mil entries
# ((84*84*4*2*1000000)*4) / (1e+6) = 225,792.0 Mb or 225 Gb

class Agent(nn.Module):

    def __init__(self, num_actions, device) -> None:
        super().__init__()
        self.device = device
        self.to(self.device)

        self._iter = 0
        self._prev_state = None

        self.num_actions = num_actions
        self._init_network(self.num_actions)
        self.target_network_update_frequency = 10000
        self._target_network_update_counter = self.target_network_update_frequency

        self.replay_buffer_size = 10000
        self._init_replay_buffer(self.replay_buffer_size)
        self.train_start_time = self.replay_buffer_size // 20
        self._train_start_counter = self.train_start_time

        self.batch_size = 32
        self.lr = 0.00025
        self.momentum = 0.95
        self.min_squared_gradient = 0.01
        self.optim = optim.RMSprop(self.network.parameters(), lr=self.lr, momentum=self.momentum, eps=self.min_squared_gradient)

        self.start_eps = 1
        self.end_eps = 0.1
        self.eps_schedule_steps = 10000
        self._init_eps_schedule(self.start_eps, self.end_eps, self.eps_schedule_steps)

        self.gamma = 0.99

    def reset(self, action, reward):
        if self._prev_state is not None:
            self._add_to_replay_buffer(self._prev_state, action, reward, None)
            self._prev_state = None

    def save(self, iter, mean_score):
        torch.save({
            "iter": iter,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "mean_score": mean_score,
        }, "agent.pt")

    def _init_network(self, num_actions):
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

    def _init_replay_buffer(self, replay_buffer_size):
        self.replay_buffer = [None for i in range(replay_buffer_size)]
        self._curr_replay_buffer_idx = 0
    
    def _add_to_replay_buffer(self, prev_state, action, reward, state):
        self.replay_buffer[self._curr_replay_buffer_idx] = (prev_state, action, reward, state)
        self._curr_replay_buffer_idx = (self._curr_replay_buffer_idx + 1) % len(self.replay_buffer)

    def _sample_from_replay_buffer(self, batch_size):
        prev_states = []
        actions = []
        rewards = []
        nonterminal_states = []
        nonterminal_states_mask = []

        nonempty_buffer_idxs = [idx for idx in range(len(self.replay_buffer)) if self.replay_buffer[idx] is not None]
        for batch_idx in range(batch_size):
            sample_idx = nonempty_buffer_idxs[torch.randint(high=len(nonempty_buffer_idxs), size=(1, )).item()]
            prev_state, action, reward, state = self.replay_buffer[sample_idx]
            prev_states.append(prev_state)
            actions.append(action)
            rewards.append(reward)
            if state is not None:
                nonterminal_states.append(state)
                nonterminal_states_mask.append(batch_idx)
        prev_states = torch.stack(prev_states)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        nonterminal_states = torch.stack(nonterminal_states)
        nonterminal_states_mask = torch.tensor(nonterminal_states_mask).to(self.device)
        batch = (prev_states, actions, rewards, nonterminal_states, nonterminal_states_mask)
        return batch

    def _init_eps_schedule(self, start_eps, end_eps, eps_schedule_steps):
        self._eps_schedule = torch.linspace(start=start_eps, end=end_eps, steps=eps_schedule_steps)
        self._curr_eps_idx = 0
        self._update_current_eps()

    def _update_current_eps(self):
        self.curr_eps = self._eps_schedule[self._curr_eps_idx]
        self._curr_eps_idx += 1

    def train(self):

        if self._train_start_counter > 0:
            return

        batch = self._sample_from_replay_buffer(self.batch_size)
        prev_states, actions, rewards, nonterminal_states, nonterminal_states_mask = batch

        qs = torch.zeros((self.batch_size, )).to(self.device)
        qs[nonterminal_states_mask] = torch.max(self.target_network(nonterminal_states), dim=-1).values
        targets = rewards + self.gamma*qs

        prev_qs = self.network(prev_states)[torch.arange(self.batch_size), actions]

        self.optim.zero_grad()
        loss = torch.mean((targets-prev_qs)**2)
        loss.backward()
        self.optim.step()

        self._target_network_update_counter -= 1
        if self._target_network_update_counter == 0:
            self.target_network = copy.deepcopy(self.network)
            self._target_network_update_counter = self.target_network_update_frequency

    def forward(self, state, action, reward):

        self._iter += 1
        
        if self._prev_state is not None:
            self._add_to_replay_buffer(self._prev_state, action, reward, state)
        self._prev_state = state

        if self._train_start_counter > 0:
            self._train_start_counter -= 1
            return torch.randint(high=self.num_actions, size=(1, )).item()
        
        if torch.bernoulli(self.curr_eps) == 1:
            return torch.randint(high=self.num_actions, size=(1, )).item()
        else:
            qs = self.network(state.unsqueeze(0))
            action = torch.argmax(qs, -1).squeeze(0).item()
        self._update_current_eps()
        return action

if __name__ == "__main__":
    replay_size = 10000
    replay_mem_size = ((84*84*4*2*replay_size)*4) / (1e+6)
    print(replay_mem_size)