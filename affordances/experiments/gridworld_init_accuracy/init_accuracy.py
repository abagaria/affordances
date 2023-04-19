"""Measure the accuracy of the initiation sets of a set of options."""

import os
import ipdb
import argparse
import functools
import numpy as np
from typing import Tuple
from collections import defaultdict

from affordances.utils import utils
from affordances.agent.hrl.option import Option
from affordances.utils.utils import safe_zip_write
from affordances.domains.visgrid import environment_builder
from affordances.experiments.gridworld_init_accuracy.options import AgentOverOptions
from affordances.utils.init_accuracy_plotting import visualize_initiation_table


def get_valid_states(env):
  """Generate the next state from which to measure accuracy."""
  positions = []
  for i in range(env.rows):
    for j in range(env.cols):
      if not env.grid.has_wall((i, j)):
        positions.append({'player_pos': (i, j)})
  return positions


def reset(env, state) -> Tuple[np.ndarray, dict]:
  """Reset simulator to the specified state."""
  return env.reset(pos=state['player_pos'])


def is_init_true(env, option, obs, info, state, method='learned') -> bool:
  """Ask the option's initiation classifier if it is available at state."""
  if method == 'learned':
    return option.optimistic_is_init_true(obs, info)
  return env.can_execute_action(state, option)  # ground-truth


def rollout_option(env, option, obs, info, method='learned') -> bool:
  """Execute option policy and report whether the option was successful."""
  
  if method == 'ground-truth':
    assert isinstance(info, dict)
    assert 'player_pos' in info
    action = option.option_idx if isinstance(option, Option) else option
    return env.can_execute_action(info, action)
  
  assert isinstance(option, Option)

  next_obs, next_info, reward, reached, n_steps, traj = option.rollout(
    env, obs, info)
  
  pos = info['player_pos']
  next_pos = next_info['player_pos']
  print(f'{option} s:{pos} sp: {next_pos} reward={reward} success={reached}')
  
  return next_obs, next_info, reached, traj


def save_data_tables(episode, ground_truth_table, measured_table):
  f1 = f'{g_log_dir}/episode_{episode}_gt.pkl'
  f2 = f'{g_log_dir}/episode_{episode}_measured.pkl'
  
  safe_zip_write(f2, measured_table)
  safe_zip_write(f1, ground_truth_table)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--environment_name', type=str, default='visgrid-6x6')
  parser.add_argument('--gestation_period', type=int, default=5)
  parser.add_argument('--timeout', type=int, default=50)
  parser.add_argument('--gpu_id', type=int, default=0)
  parser.add_argument('--n_episodes', type=int, default=101)
  parser.add_argument('--plotting_frequency', type=int, default=-1)
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  parser.add_argument('--plot_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_plots')
  parser.add_argument('--gridsize', type=int, default=6)
  parser.add_argument('--use_random_maze', action='store_true', default=False)
  args = parser.parse_args()

  assert args.gridsize in (6, 13)
  assert args.environment_name in ("visgrid-6x6", "visgrid-13x13")

  g_log_dir = os.path.join(args.log_dir, args.experiment_name)
  g_plot_dir = os.path.join(args.plot_dir, args.experiment_name)

  utils.create_log_dir(args.log_dir)
  utils.create_log_dir(args.plot_dir)
  utils.create_log_dir(g_log_dir)
  utils.create_log_dir(g_plot_dir)
  utils.set_random_seed(args.seed)

  # Start with an open gridworld
  environment = environment_builder(use_random_maze=args.use_random_maze,
                                    size=args.gridsize,
                                    seed=1, test=False)
  obs0, info0 = environment.reset()

  # s -> o -> measured
  measured_initiations = defaultdict(functools.partial(defaultdict, list))
  ground_truth_initiations = defaultdict(functools.partial(defaultdict, list))

  agent_over_options = AgentOverOptions(
    environment, gestation_period=args.gestation_period,
    timeout=args.timeout, gpu=args.gpu_id,
    image_dim=obs0.squeeze().shape[0]
  )
  options = agent_over_options.options

  for episode in range(args.n_episodes):

    print("*" * 80)
    print("Episode ", episode)
    print("*" * 80)

    for state in get_valid_states(environment):

      for option in options:

        # Reset the simulator to the specific state
        obs, info = reset(environment, state)

        # Measured option availability
        measured_init = is_init_true(environment, option, obs, info, state)
        measured_initiations[state["player_pos"]][str(option)].append(measured_init)

        # Ground-truth monte-carlo rollout
        next_obs, next_info, reached, traj = rollout_option(environment, option, obs, info)
        ground_truth_initiations[state["player_pos"]][str(option)].append(reached)

        # On-policy things: add data to policy, classifier and GVF buffers;
        # perform updates to the option policy. Reserved for available options.
        if measured_init:
          option.update_option_after_rollout(traj, reached)

    # Off-policy things: minibatch updates on GVF and all initiation classifiers 
    agent_over_options.update_initiation_gvf(episode_duration=100)
    
    for option in options:
      option.update_option_initiation_classifiers()
    
    if episode % 10 == 0:
      save_data_tables(episode, ground_truth_initiations, measured_initiations)
      visualize_initiation_table(ground_truth_initiations, options, g_plot_dir, 'gt', episode)
      visualize_initiation_table(measured_initiations, options, g_plot_dir, 'measured', episode)
