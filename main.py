import gymnasium as gym
import ale_py
from agent import Agent
import torch
import time
import datetime
import numpy as np
import os
import argparse

def print_and_log(string: str, log_file_path: str | os.PathLike | None) -> None:
    log_string = f"{datetime.datetime.now()}: {string}"
    print(log_string)
    if log_file_path:
        with open(log_file_path, "a") as f:
            f.write(log_string + "\n")

def train(agent: Agent, env: gym.Env, num_train_iters:int = 25000000, num_eval_iters:int = 25000, train_freq:int = 4, eval_freq:int = 2500, use_checkpoint:bool = True, checkpoint_freq: int = 12500, log_file_path: str | os.PathLike = None, rewards_file_path: str | os.PathLike = "eval_mean_rewards_per_episode.npy") -> None:

    agent.train()
    start_iter = 0
    episode_num = 1
    eval_mean_rewards_per_episode = []
    if use_checkpoint:
        if agent.has_checkpoint():
            print_and_log(f": Found checkpoint. Loading...", log_file_path=log_file_path)
            start_iter, episode_num = agent.load_checkpoint()
            print_and_log(f"Continuing from iteration {start_iter}", log_file_path=log_file_path)
        if os.path.exists(rewards_file_path):
            with open(rewards_file_path, "rb") as f:
                eval_mean_rewards_per_episode = np.load(f).tolist()

    observation, info = env.reset()
    action, reward = (0, 0)
    episode_reward = 0
    episode_start_time = time.time()
    for iter in range(start_iter, num_train_iters):

        action = agent(observation, action, reward)

        if (iter % train_freq) == 0:
            agent.train_step()

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if use_checkpoint and ((iter % checkpoint_freq) == 0):
            print_and_log(f"Saving checkpoint at iteration {iter}...", log_file_path=log_file_path)
            agent.save_checkpoint()

        if terminated or truncated:
            episode_end_time = time.time()
            episode_time = episode_end_time - episode_start_time
            episode_log_string = f"episode_num: {episode_num}, episode_time: {episode_time:.2f}s, episode reward: {episode_reward:.2f}, iter: {iter}"
            print_and_log(episode_log_string, log_file_path=log_file_path)

            episode_num += 1
            episode_reward = 0

        if (iter % eval_freq) == 0:
            print_and_log(f"Iteration {iter} reached, evaluating...", log_file_path=log_file_path)
            mean_reward_per_episode = eval(agent, env, num_iters=num_eval_iters, log_file_path=log_file_path, summarise_episodes=False)
            
            eval_mean_rewards_per_episode.append(mean_reward_per_episode)
            
            with open(rewards_file_path, "wb") as f:
                np.save(f, np.array(eval_mean_rewards_per_episode))

            print_and_log("Saving evaluation results...", log_file_path=log_file_path)

            agent.train()

        if terminated or truncated or ((iter % eval_freq) == 0):
            agent.reset(action, reward)
            observation, info = env.reset()
            action, reward = (0, 0)
            episode_start_time = time.time()

def eval(agent: Agent, env: gym.Env, num_iters: int = 100000, use_checkpoint: bool = True, log_file_path: str | os.PathLike = None, summarise_episodes: bool = True) -> np.ndarray:

    agent.eval()
    if use_checkpoint and agent.has_checkpoint():
        agent.load_checkpoint()

    total_reward = 0

    with torch.no_grad():

        observation, info = env.reset()
        action, reward = (0, 0)
        episode_num = 1
        episode_reward = 0
        episode_start_time = time.time()
        for iter in range(num_iters):

            action = agent(observation, action, reward)

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            total_reward += reward

            if terminated or truncated:
                episode_end_time = time.time()
                episode_time = episode_end_time - episode_start_time
                if summarise_episodes:
                    episode_log_string = f"episode_num: {episode_num}, episode_time: {episode_time:.2f}s, episode reward: {episode_reward:.2f}, iter: {iter}"
                    print_and_log(episode_log_string, log_file_path=log_file_path)

                episode_num += 1
                episode_reward = 0
                agent.reset(action, reward)
                observation, info = env.reset()
                action, reward = (0, 0)
                episode_start_time = time.time()

        mean_reward_per_episode = total_reward / episode_num
        eval_string = f"Evaluation complete. Obtained mean reward of {mean_reward_per_episode:.2f} across {episode_num} episodes"
        print_and_log(eval_string, log_file_path=log_file_path)
        return mean_reward_per_episode
    
def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="evaluate pre-trained agent")
    parser.add_argument("--render", action="store_true", help="render environment")
    args = parser.parse_args()
    train_mode = not args.eval
    should_render = args.render

    torch.manual_seed(2024)
    gym.register_envs(ale_py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default v4 env doesn't handle frame flickering, so use gym wrappers to fix
    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human" if (not train_mode) or should_render else None)
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = gym.wrappers.ClipReward(env, min_reward=-1, max_reward=1)
    env = gym.wrappers.NumpyToTorch(env, device=device)

    agent = Agent(num_actions=env.action_space.n.item(), device=device)

    
    if train_mode:
        train(agent, env, log_file_path="train_log.txt")
    else:
        eval(agent, env, log_file_path="eval_log.txt")
    env.close()

if __name__ == "__main__":
    main()
