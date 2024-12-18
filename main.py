import gymnasium as gym
import ale_py
from agent import Agent
import torch
import datetime
gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default v4 env doesn't handle frame flickering, so use gym wrappers to fix
env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)
env = gym.wrappers.NumpyToTorch(env, device=device)
agent = Agent(num_actions=env.action_space.n.item(), device=device)

num_iters = 10000000
agent_train_freq = 4
observation, info = env.reset()
action, reward = (0, 0)
episode_num = 1
total_score = 0
score = 0
for iter in range(num_iters):

    action = agent(observation, action, reward)
    if (iter % agent_train_freq) == 0:
        agent.train()

    observation, reward, terminated, truncated, info = env.step(action)
    score += reward

    if terminated or truncated:

        total_score += score
        mean_score = total_score / episode_num
        log_str = f"{datetime.datetime.now()}, episode_num: {episode_num}, episode score: {score}, iter: {iter}, mean_score: {mean_score}"
        print(log_str)
        with open("log.txt", "a") as f:
            f.write(log_str + "\n")
        if (iter % 10000) == 0:
            agent.save(iter, mean_score)

        score = 0
        episode_num += 1

        agent.reset(action, reward)
        observation, info = env.reset()
        action, reward = (0, 0)

env.close()
