import os
import random
import argparse
import numpy as np

from affordances.utils import utils
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.domains.minigrid import environment_builder, determine_goal_pos


def create_agent(
  n_actions,
  gpu,
  n_input_channels,
  env_steps=50_000,
  lr=6.25e-5,
  sigma=0.5
):
    kwargs = dict(
      n_atoms=51, v_max=10., v_min=-10.,
      noisy_net_sigma=sigma, lr=lr, n_steps=3,
      betasteps=env_steps // 4,
      replay_start_size=1024, 
      replay_buffer_size=int(3e5),
      gpu=gpu, n_obs_channels=2*n_input_channels,
      use_custom_batch_states=False,
      epsilon=0.1
    )
    return Rainbow(n_actions, **kwargs)


def concat(obs, goal):
   return np.concatenate((obs, goal), axis=0)


def rollout(agent: Rainbow, env, goal: np.ndarray):
  global goal_observation

  done = False
  episode_reward = 0.
  obs, info = env.reset()
  trajectory = []
  while not done:
    action = agent.act(concat(obs, goal))
    next_obs, reward, done, info = env.step(action)
    trajectory.append((obs, action, reward, next_obs, info))

    if reward == 1:
      goal_observation = next_obs

    obs = next_obs
    episode_reward += reward
  return trajectory


def experience_replay(agent: Rainbow, transitions, goal, goal_info):
  relabeled_trajectory = []
  for state, action, _, next_state, info in transitions:
    sg = concat(state, goal)
    nsg = concat(next_state, goal)
    reached = info['player_pos'] == goal_info['player_pos']
    reward = float(reached)
    relabeled_trajectory.append((
      sg, action, reward, nsg, reached, info['needs_reset']))
    if reached:
      break
  agent.experience_replay(relabeled_trajectory)


def select_goal(task_goal, task_goal_info, method='task'):
  if method == 'task':
    return task_goal, task_goal_info
  raise NotImplementedError


def pick_hindsight_goal(transitions, method='final'):
  if method == 'final':
    goal_transition = transitions[-1]
    goal = goal_transition[-2]
    goal_info = goal_transition[-1]
    return goal, goal_info
  if method == 'future':
    start_idx = len(transitions) // 2
    goal_idx = random.randint(start_idx, len(transitions) - 1)
    goal_transition = transitions[goal_idx]
    goal = goal_transition[-2]
    goal_info = goal_transition[-1]
    return goal, goal_info
  raise NotImplementedError(method)


def train(agent: Rainbow, env, n_episodes,
          task_goal: np.ndarray, task_goal_info: dict):
  for _ in range(n_episodes):
    goal, goal_info = select_goal(task_goal, task_goal_info)
    trajectory = rollout(agent, env, goal)
    experience_replay(agent, trajectory, goal, goal_info)
    hindsight_goal, hindsight_goal_info = pick_hindsight_goal(trajectory)
    experience_replay(agent, trajectory, hindsight_goal, hindsight_goal_info)


def evaluate(agent: Rainbow, env, task_goal: np.ndarray, n_episodes: int = 10):
  rewards = []
  for _ in range(n_episodes):
    trajectory = rollout(agent, env, task_goal)
    episode_rewards = [trans[2] for trans in trajectory]
    rewards.append(sum(episode_rewards))
  return np.mean(rewards)


def log(msr):
  global success_rates
  success_rates.append(msr)
  fname = f'{g_log_dir}/log_seed{args.seed}.pkl'
  utils.safe_zip_write(fname, {'success_rates': success_rates})


def run(agent: Rainbow, env, n_iterations,
        task_goal: np.ndarray, task_goal_info: dict):
  for iter in range(n_iterations):
    train(agent, env, 10, task_goal, task_goal_info)
    msr = evaluate(agent, env, task_goal, n_episodes=10)
    print(f'[EvaluationIter={iter}] Mean Success Rate: {msr}')
    log(msr)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--sub_dir', type=str, default='', help='sub dir for sweeps')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--environment_name', type=str, default='MiniGrid-Empty-8x8-v0')
  parser.add_argument('--n_iterations', type=int, default=1000)
  parser.add_argument('--lr', type=float, default=6.25e-5)
  parser.add_argument('--sigma', type=float, default=0.5)
  parser.add_argument('--bonus_scale', type=float, default=0)
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  args = parser.parse_args()

  g_log_dir = os.path.join(args.log_dir, args.experiment_name, args.sub_dir)

  utils.create_log_dir(args.log_dir)
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name))
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name, args.sub_dir))
  utils.create_log_dir(g_log_dir)

  utils.set_random_seed(args.seed)

  environment = environment_builder(
    args.environment_name, exploration_reward_scale=args.bonus_scale)

  rainbow_agent = create_agent(
    environment.action_space.n,
    gpu=args.gpu,
    n_input_channels=1,
    lr=args.lr,
    sigma=args.sigma
  )

  s0, info0 = environment.reset()

  goal_info_dict = dict(player_pos=determine_goal_pos(environment))
  goal_observation = np.zeros_like(s0)

  success_rates = []

  run(
    rainbow_agent,
    environment,
    args.n_iterations,
    goal_observation,
    goal_info_dict
  )
