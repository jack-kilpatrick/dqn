import torch
from torch import nn, optim
import gymnasium as gym
import copy
import os

class Agent(nn.Module):
    """Base agent class"""
    
    def __init__(self, name:str, env: gym.Env, device: torch.device) -> None:
        "Initialise an agent, setting the name and environment name and moving the module to the specified device"
        super().__init__()
        self.name = name
        self.env_name = env.unwrapped.spec.id
        self.device = device
        self.to(self.device)
        
    def train_mode(self) -> None:
        """Set relevant internal attributes to training mode e.g. for maintaining a replay buffer or updating network batchnorm layers"""
        raise NotImplementedError

    def eval_mode(self) -> None:
        """Set relevant internal atttributes to training mode e.g. disabling training exploration strategy"""
        raise NotImplementedError

    def reset(self, reward: float) -> None:
        """Perform any internal operations needed at the end of an episode"""
        raise NotImplementedError
    
    def has_checkpoint(self) -> None:
        """Return true if the necessary agent checkpoint files are present"""
        raise NotImplementedError

    def save_checkpoint(self, iter: int, episode_num: int, best_mean_reward_per_episode: float|None = None) -> None:
        """Save necessary agent checkpoint files, mark the current iteration, episode number and best mean episodic reward for training resumption"""
        raise NotImplementedError

    def load_checkpoint(self) -> tuple[int, int, int]:
        """Load necessary agent checkpoint files, return the iteration, episode and best mean episode reward achieved during training"""
        raise NotImplementedError

    def train_step(self) -> None:
        """Run one step of the agents training process"""
        raise NotImplementedError

    def forward(self, state: torch.Tensor, reward: float) -> torch.Tensor:
        """Predict an action given the current state. Also given the previous timestep reward as per typical MDP conventions"""
        raise NotImplementedError

class DQN(Agent):

    def __init__(self, name: str, env: gym.Env, device: torch.device) -> None:
        "Initialise a DQN agent with the default network and hyperparameters set according to the original paper"
        super().__init__(name, env, device)

        self._init_checkpoint_paths()

        self._iter = 0
        self._episode_num = 1
        self._prev_state = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._init_network(self.observation_space, self.action_space)
        self.target_network_update_frequency = 10000
        self._target_network_update_counter = 0

        self.replay_buffer_size = 1000000
        self.save_replay_buffer = True
        self._init_replay_buffer(self.replay_buffer_size)
        self.train_start_time = 12500
        self._train_start_counter = 0

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

    def train_mode(self) -> None:
        """Set the module to training mode"""
        if not self.training:
            self.train()

    def eval_mode(self) -> None:
        """Set the module to evaluation mode"""
        if self.training:
            self.eval()

    def reset(self, reward: float) -> None:
        """Add the episode terminal transition to the replay buffer"""
        if self.training:
            if self._prev_state is not None:
                self._add_to_replay_buffer(self._prev_state, self._prev_action, reward, None)
                self._prev_state = None
            self._episode_num += 1

    def _init_checkpoint_paths(self) -> None:
        """Setup checkpoint paths for both the agent and replay buffer"""
        self.checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", self.env_name, self.name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self._agent_checkpoint_path = os.path.join(self.checkpoint_dir, "agent.pt")
        self._replay_buffer_checkpoint_path = os.path.join(self.checkpoint_dir, "replay_buffer.pt")
        self.checkpoint_paths = [self._agent_checkpoint_path, self._replay_buffer_checkpoint_path]

    def has_checkpoint(self) -> None:
        """Check if the agent checkpoint exists. The replay buffer is not considered here since it will be reinitialised if unavailable"""
        return os.path.exists(self._agent_checkpoint_path)

    def save_checkpoint(self, iter: int, episode_num: int, best_mean_reward_per_episode: float|None = None) -> None:
        """Save the agent and replay buffer checkpoint"""
        torch.save({
            "iter": iter,
            "episode_num": episode_num,
            "best_mean_reward_per_episode": best_mean_reward_per_episode,
            "target_network_update_counter": self._target_network_update_counter,
            "train_start_counter": self._train_start_counter,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
        }, self._agent_checkpoint_path)
        if self.save_replay_buffer:
            torch.save({
                "curr_replay_buffer_idx": self._curr_replay_buffer_idx,
                "replay_buffer_num_elts": self._replay_buffer_num_elts,
                "replay_buffer": self.replay_buffer,
            }, self._replay_buffer_checkpoint_path)

    def load_checkpoint(self) -> tuple[int, int, int]:
        """Load the agent checkpoint. Attempt to load the replay buffer checkpoint if in training mode, if missing or corrupted refill before training resumption"""
        checkpoint = torch.load(
            self._agent_checkpoint_path,
            weights_only=True,
        )
        best_mean_reward_per_episode = checkpoint.get("best_mean_reward_per_episode")
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
        self._iter = checkpoint["iter"]
        self._episode_num = checkpoint["episode_num"]
        self._target_network_update_counter = checkpoint["target_network_update_counter"]
        self._train_start_counter = checkpoint["train_start_counter"]
        self._update_current_eps()

        if self.training:
            try:
                replay_buffer_checkpoint = torch.load(
                    self._replay_buffer_checkpoint_path,
                    weights_only=True,
                )
                self.replay_buffer = replay_buffer_checkpoint["replay_buffer"]
                self._curr_replay_buffer_idx = replay_buffer_checkpoint["curr_replay_buffer_idx"]
                self._replay_buffer_num_elts = replay_buffer_checkpoint["replay_buffer_num_elts"]
            except (FileNotFoundError, RuntimeError) as _:
                self._iter -= self.train_start_time
                self._train_start_counter = 0
      
        return self._iter, self._episode_num, best_mean_reward_per_episode


    def _init_network(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Discrete) -> None:
        """Build the agent Q-network. Assumes that 3D observations are images (RGB or frame-stacked) and builds a CNN in this case, otherwise builds an MLP"""
        num_actions = action_space.n
        state_contains_images = (len(observation_space.shape) == 3)
        if state_contains_images:
            state_num_channels = observation_space.shape[0]
            self.network = nn.Sequential(
                nn.Conv2d(in_channels=state_num_channels, out_channels=32, kernel_size=8, stride=4),
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
        else:
            state_num_elts = torch.prod(torch.tensor(observation_space.shape)).int().item()
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(state_num_elts, 400),
                nn.ReLU(),
                nn.Linear(400, 300),
                nn.ReLU(),
                nn.Linear(300, num_actions)
            ).to(self.device)
        self.target_network = copy.deepcopy(self.network)

    def _init_replay_buffer(self, replay_buffer_size: int) -> None:
        """Initialise the replay buffer, and internal state to track the position of the next element and number of contained elements"""
        self.replay_buffer = [None for _ in range(replay_buffer_size)]
        self._curr_replay_buffer_idx = 0
        self._replay_buffer_num_elts = 0
    
    def _add_to_replay_buffer(self, prev_state: torch.Tensor, action: int, reward: float, state: torch.Tensor) -> None:
        """Add one transition (prev_state, action, reward, state) to the replay buffer. Cast and move to save memory. Updates internal buffer counters"""
        prev_state = (prev_state * 255).to(device=torch.device("cpu"), dtype=torch.uint8)
        action = action
        reward = reward
        state = None if state is None else (state * 255).to(device=torch.device("cpu"), dtype=torch.uint8)

        self.replay_buffer[self._curr_replay_buffer_idx] = (prev_state, action, reward, state)
        self._curr_replay_buffer_idx = (self._curr_replay_buffer_idx + 1) % self.replay_buffer_size
        if self._replay_buffer_num_elts < self.replay_buffer_size:
            self._replay_buffer_num_elts += 1

    def _sample_from_replay_buffer(self, batch_size: int) -> None:
        """Uniformly sample a mini-batch of transitions from the replay buffer"""
        prev_states = []
        actions = []
        rewards = []
        nonterminal_states = []
        nonterminal_states_mask = []

        for batch_idx, sample_idx in enumerate(torch.randint(high=self._replay_buffer_num_elts, size=(batch_size, ))):
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
    
    def _get_curr_eps(self) -> torch.Tensor:
        """Return current epsilon value used for exploration"""
        return self._curr_eps if self.training else torch.tensor(self.eval_eps)

    def _init_eps_schedule(self, start_eps: float, end_eps: float, eps_schedule_steps: int) -> None:
        """Initialise linear annealing schedule for epsilon value used for exploration"""
        self._eps_schedule = torch.linspace(start=start_eps, end=end_eps, steps=eps_schedule_steps)
        self._update_current_eps()

    def _update_current_eps(self) -> None:
        """Update epsilon value used for exploration according to internal linear annealing schedule"""
        if self._iter < self.eps_schedule_steps:
            self._curr_eps = self._eps_schedule[self._iter]
        else:
            self._curr_eps = self._eps_schedule[-1]

    def train_step(self) -> None:
        """Using a replay minibatch, perform one optimisation step on the Smooth L1 Loss of the predicted Q-values against the target Q-values"""
        if self._train_start_counter < self.train_start_time:
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

    def forward(self, state: torch.Tensor, reward: float) -> torch.Tensor:
        """
        Predict the Q-values of each possible action for the current state.
        Select the maximising action with probability 1-epsilon and a random action otherwise.
        Also pick a random action if the replay buffer has yet to be sufficiently filled according to the associated internal counters.
        """
        if self.training:
            if self._prev_state is not None:
                self._add_to_replay_buffer(self._prev_state, self._prev_action, reward, state)
            self._prev_state = state
        
        if (self._train_start_counter < self.train_start_time and self._iter < self.train_start_time) or torch.bernoulli(self._get_curr_eps()) == 1:
            action = torch.randint(high=self.action_space.n, size=(1, )).item()
        else:
            qs = self.network(state.unsqueeze(0))
            action = torch.argmax(qs, -1).squeeze(0)
            
        if self.training:
            self._iter += 1
            self._update_current_eps()
            if self._train_start_counter < self.train_start_time:
                self._train_start_counter += 1
            self._prev_action = action
            
        return action