"""Measure the accuracy of the initiation sets of a set of options."""

import os
import ipdb
import copy
import pickle
import argparse
import functools
import numpy as np
from typing import Tuple
from collections import defaultdict

from affordances.utils import utils
from affordances.agent.hrl.option import Option
from affordances.utils.utils import safe_zip_write
from affordances.domains.montezuma.montezuma import environment_builder, montezuma_subgoals
from affordances.experiments.gridworld_init_accuracy.options import AgentOverOptions
from affordances.utils.init_accuracy_plotting import visualize_initiation_table
from affordances.utils.plotting import visualize_initiation_classifier


path_to_resources = os.path.expanduser("~/Downloads/")

def get_valid_states(env):
  states = []
  for fname in montezuma_subgoals:
    with open(os.path.join(path_to_resources, fname), 'rb') as f:
      state_dict = pickle.load(f)
      states.append(
        dict(
          player_pos=state_dict['position'],
          ram=state_dict['ram'],
          state=state_dict['state']
        )
      )
  return states


def reset(env, state) -> Tuple[np.ndarray, dict]:
  """Reset simulator to the specified state."""
  return env.reset(ram=state['ram'], state=state['state'])


def rollout_option(env, option, obs, info, method='learned') -> bool:
  """Execute option policy and report whether the option was successful."""
  assert isinstance(option, Option)
  
  next_obs, next_info, reward, reached, n_steps, traj = option.rollout(
    env, obs, info)
  
  print(f'{option} g:{option.subgoal_info["player_pos"]} s:{info["player_pos"]}',
        f'sp: {next_info["player_pos"]} T={n_steps} R={reward} success={reached}')
  
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
  parser.add_argument('--gestation_period', type=int, default=5)
  parser.add_argument('--timeout', type=int, default=50)
  parser.add_argument('--gpu_id', type=int, default=0)
  parser.add_argument('--n_episodes', type=int, default=101)
  parser.add_argument('--plotting_frequency', type=int, default=-1)
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  parser.add_argument('--plot_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_plots')
  parser.add_argument('--max_frames_per_episode', type=int, default=30*60*60)
  args = parser.parse_args()

  g_log_dir = os.path.join(args.log_dir, args.experiment_name)
  g_plot_dir = os.path.join(args.plot_dir, args.experiment_name)

  utils.create_log_dir(args.log_dir)
  utils.create_log_dir(args.plot_dir)
  utils.create_log_dir(g_log_dir)
  utils.create_log_dir(g_plot_dir)
  utils.set_random_seed(args.seed)

  # Create a monte env
  environment = environment_builder(max_frames_per_episode=args.max_frames_per_episode)
  obs0, info0 = environment.reset()

  # s -> o -> measured
  measured_initiations = defaultdict(functools.partial(defaultdict, list))
  ground_truth_initiations = defaultdict(functools.partial(defaultdict, list))

  states = get_valid_states(environment)

  # Create subgoals and options for MR
  agent_over_options = AgentOverOptions(
    environment, gestation_period=args.gestation_period,
    timeout=args.timeout, gpu=args.gpu_id,
    image_dim=obs0._frames[0].squeeze().shape[0],
    rams=states
  )
  options = agent_over_options.options

  for episode in range(args.n_episodes):

    print("*" * 80)
    print("Episode ", episode)
    print("*" * 80)

    for state in states:

      for option in options:
        assert isinstance(option, Option)

        # Reset the simulator to the specific state
        obs, info = reset(environment, state)

        # If the state is in the option's goal, move on to the next option
        if option.is_term_true(obs, info):
          continue

        # Measured option availability
        measured_init = option.optimistic_is_init_true(obs, info)
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
      [visualize_initiation_classifier(option.initiation_classifier,
                                      agent_over_options.initiation_gvf,
                                      agent_over_options.initiation_gvf.initiation_replay_buffer,
                                      option.subgoal_obs, option.subgoal_info,
                                      str(option), episode, args.plot_dir,
                                      args.experiment_name, args.seed) for option in options]
