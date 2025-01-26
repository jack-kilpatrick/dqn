import gymnasium as gym
from matplotlib import pyplot as plt
from tqdm import tqdm
from agent import Agent, DQN
import torch
import time
import datetime
import numpy as np
import os
import argparse

def print_and_log(string: str, log_file_path: str | os.PathLike | None, timestamp: bool = True) -> None:
    """Print a string and write it to the given file in append mode. Optionally timestamp it (done by default)."""
    log_string = f"{datetime.datetime.now()}: {string}"
    print(log_string)
    if log_file_path:
        with open(log_file_path, "a") as f:
            f.write(log_string + "\n")

def train(agent: Agent, env: gym.Env, num_train_iters:int = 50000000, train_freq:int = 4, num_eval_iters:int = 125000, eval_freq:int = 250000, log_file_path: str | os.PathLike = None, eval_rewards_file_path: str | os.PathLike | None = None, eval_rewards_figure_file_path: str | os.PathLike | None = None) -> Agent:
    """
    Train an agent in the given environment for a specified number of iterations.
    Evaluate throughout training in a separate loop at the specified frequency and for the specified duration.
    Optionally checkpoint the best agent according to the evaluations (done by default). 
    Returns the trained agent.
    """
    agent.train_mode()
    start_iter = 0
    episode_num = 1
    eval_mean_rewards_per_episode = []
    best_mean_reward_per_episode = -np.inf
    if agent.has_checkpoint():
        print_and_log(f": Found checkpoint. Loading...", log_file_path=log_file_path)
        start_iter, episode_num, best_mean_reward_per_episode = agent.load_checkpoint()
        if best_mean_reward_per_episode is None:
            best_mean_reward_per_episode = -np.inf
        print_and_log(f"Continuing from iteration {start_iter}", log_file_path=log_file_path)
        if os.path.exists(eval_rewards_file_path):
            with open(eval_rewards_file_path, "rb") as f:
                eval_mean_rewards_per_episode = np.load(f).tolist()

    observation, info = env.reset()
    reward = 0
    episode_reward = 0
    episode_start_time = time.time()
    for iter in tqdm(range(start_iter, num_train_iters), initial=0, desc="Training"):

        action = agent(observation, reward)

        is_train_iter = ((iter % train_freq) == 0)
        if is_train_iter:
            agent.train_step()

        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            episode_end_time = time.time()
            episode_time = episode_end_time - episode_start_time
            episode_log_string = f"episode_num: {episode_num}, episode_time: {episode_time:.2f}s, episode reward: {episode_reward:.2f}, iter: {iter}"
            print_and_log(episode_log_string, log_file_path=log_file_path)

            episode_num += 1
            episode_reward = 0

        is_eval_iter = ((iter % eval_freq) == 0)
        if is_eval_iter:
            print_and_log(f"Iteration {iter} reached, evaluating...", log_file_path=log_file_path)

            agent.eval_mode()
            mean_reward_per_episode = eval(agent, env, num_iters=num_eval_iters, log_file_path=log_file_path, summarise_episodes=False)
            agent.train_mode()

            eval_mean_rewards_per_episode.append(mean_reward_per_episode)
            if eval_rewards_file_path:
                print_and_log("Saving evaluation results...", log_file_path=log_file_path)
                with open(eval_rewards_file_path, "wb") as f:
                    np.save(f, np.array(eval_mean_rewards_per_episode))
                if eval_rewards_figure_file_path:
                    plt.plot(np.arange(len(eval_mean_rewards_per_episode)), eval_mean_rewards_per_episode)
                    plt.ylabel("Average score per episode")
                    plt.xlabel(f"Training epochs")
                    plt.savefig(eval_rewards_figure_file_path)
                    plt.close()

            if mean_reward_per_episode > best_mean_reward_per_episode:
                prev_best_mean_reward_per_episode = best_mean_reward_per_episode
                best_mean_reward_per_episode = mean_reward_per_episode
                print_and_log(f"New best mean reward achieved ({prev_best_mean_reward_per_episode:.2f} -> {best_mean_reward_per_episode:2f}).", log_file_path=log_file_path)
                print_and_log(f"Saving checkpoint...", log_file_path=log_file_path)
                agent.save_checkpoint(iter=iter, episode_num=episode_num, best_mean_reward_per_episode=best_mean_reward_per_episode)

        if terminated or truncated or is_eval_iter:
            agent.reset(reward)
            observation, info = env.reset()
            reward = 0
            episode_start_time = time.time()

    return agent

def eval(agent: Agent, env: gym.Env, num_iters: int = 125000, log_file_path: str | os.PathLike = None, summarise_episodes: bool = True) -> float:
    """
    Evaluate an agent in the given environment for a specified number of iterations.
    Returns the mean reward per episode across the evaluation.
    """
    agent.eval_mode()
    if agent.has_checkpoint():
        agent.load_checkpoint()

    total_reward = 0

    with torch.no_grad():

        observation, info = env.reset()
        reward = 0
        episode_num = 1
        episode_reward = 0
        episode_start_time = time.time()
        for iter in tqdm(range(num_iters), desc="Evaluation"):

            action = agent(observation, reward)

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
                agent.reset(reward)
                observation, info = env.reset()
                reward = 0
                episode_start_time = time.time()

        mean_reward_per_episode = total_reward / episode_num
        eval_string = f"Evaluation complete. Obtained mean reward of {mean_reward_per_episode:.2f} across {episode_num} episodes"
        print_and_log(eval_string, log_file_path=log_file_path)
        return mean_reward_per_episode
    
def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", help="Name of the agent")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="Name of gymnasium environment to use")
    parser.add_argument("--num_train_iters", type=int, default=50000000, help="Number of training iterations/environment steps")
    parser.add_argument("--train_freq", type=int, default=4, help="Number of iterations/environment steps per training step")
    parser.add_argument("--num_eval_iters", type=int, default=125000, help="Number of evaluation iterations/environment steps (Note: for an isolated evaluation if --eval is also passed, otherwise for intermediate evaluations in training)")
    parser.add_argument("--eval_freq",type=int, default=250000, help="Number of iterations/environment steps per intermediate evaluation in training")
    parser.add_argument("--eval", action="store_true", help="Evaluate a pre-trained agent for a given number of iterations/environment steps")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()
    
    agent_name:str = args.agent
    env_name:str = args.env
    train_mode:int = not args.eval
    num_train_iters:int = args.num_train_iters
    train_freq:int = args.train_freq
    num_eval_iters:int = args.num_eval_iters
    eval_freq:int = args.eval_freq
    render:bool = ((not train_mode) or args.render) 

    torch.manual_seed(2024)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = os.path.join(os.getcwd(), "logs", env_name, agent_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    videos_dir = os.path.join(log_dir, "videos")
    if not os.path.isdir(videos_dir):
        os.makedirs(videos_dir)
    metrics_dir = os.path.join(log_dir, "metrics")
    if not os.path.isdir(metrics_dir):
        os.makedirs(metrics_dir)
    figures_dir = os.path.join(log_dir, "figures")
    if not os.path.isdir(figures_dir):
        os.makedirs(figures_dir)

    train_log_file_path = os.path.join(log_dir, "train_log.txt")
    eval_log_file_path = os.path.join(log_dir, "eval_log.txt")
    eval_rewards_file_path = os.path.join(metrics_dir, "eval_mean_rewards_per_episode.npy")
    eval_rewards_figure_file_path = os.path.join(figures_dir, "eval_mean_rewards_per_episode.png")

    env_in_default_registry = (env_name in gym.registry)
    import ale_py
    gym.register_envs(ale_py)
    env_in_ale_registry = (env_name in gym.registry)
    is_atari_env = env_in_ale_registry and not env_in_default_registry
    
    env = gym.make(env_name, render_mode="human" if render else "rgb_array")
    if not render:
        env = gym.wrappers.RecordVideo(env, fps=60, name_prefix=f"unwrapped-env", video_folder=videos_dir, step_trigger=lambda step:(step % (eval_freq * 4 if is_atari_env else eval_freq) == 0))
    if is_atari_env:
        env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env = gym.wrappers.ClipReward(env, min_reward=-1, max_reward=1)
    env = gym.wrappers.NumpyToTorch(env, device=device)
    env: gym.Env = env

    agent = DQN(name=agent_name, env=env, device=device)

    if train_mode:
        train(agent, env, num_train_iters=num_train_iters, train_freq=train_freq, num_eval_iters=num_eval_iters, eval_freq=eval_freq, log_file_path=train_log_file_path, eval_rewards_file_path=eval_rewards_file_path, eval_rewards_figure_file_path=eval_rewards_figure_file_path)
    else:
        eval(agent, env, num_iters=num_eval_iters, log_file_path=eval_log_file_path)
    env.close()

if __name__ == "__main__":
    main()
