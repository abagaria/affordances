"""Measure the accuracy of the initiation sets of a set of options."""

import os
import pickle
import argparse
import functools
import numpy as np
from typing import Tuple
from collections import defaultdict

from affordances.utils import utils
from affordances.agent.hrl.option import Option
from affordances.utils.utils import safe_zip_write
from affordances.domains.minigrid import environment_builder, get_open_grid_positions
from affordances.domains.minigrid import four_rooms_get_room_centers, minigrid_open_grid_subgoals

from affordances.experiments.gridworld_init_accuracy.options import AgentOverOptions
from affordances.utils.init_accuracy_plotting import visualize_initiation_table
from affordances.utils.plotting import visualize_initiation_classifier
from affordances.utils.plotting import visualize_gc_initiation_learner


def get_valid_states(env):
  states = []
  for pos in get_open_grid_positions(env):
    states.append(
      dict(
        player_pos=pos
      )
    )
  return states


def get_subgoals(env):
  states = []
  if args.level_name == 'four-rooms':
    subgoals = four_rooms_get_room_centers()
  else:
    assert args.level_name == 'open-grid', args.level_name
    subgoals = minigrid_open_grid_subgoals()
  for pos in subgoals:
    states.append(
      dict(
        player_pos=pos
      )
    )
  return states


def reset(env, state) -> Tuple[np.ndarray, dict]:
  """Reset simulator to the specified state."""
  return env.reset_to(
    state['player_pos'],
    randomize_direction=args.randomize_direction
  )


def rollout_option(env, option, obs, info, method='learned') -> bool:
  """Execute option policy and report whether the option was successful."""
  assert isinstance(option, Option)
  
  next_obs, next_info, reward, reached, n_steps, traj = option.rollout(
    env, obs, info)
  
  print(f'{option} I={measured_init} g:{option.subgoal_info["player_pos"]}',
        f's:{info["player_pos"]} sp: {next_info["player_pos"]} T={n_steps}',
        f'R={reward} success={reached}')
  
  return next_obs, next_info, reached, traj


def save_data_tables(episode, ground_truth_table, measured_table):
  f1 = f'{g_log_dir}/episode_{episode}_gt_seed{args.seed}.pkl'
  f2 = f'{g_log_dir}/episode_{episode}_measured_seed{args.seed}.pkl'
  
  safe_zip_write(f2, measured_table)
  safe_zip_write(f1, ground_truth_table)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--sub_dir', type=str, default='', help='sub dir for sweeps')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--gestation_period', type=int, default=10)
  parser.add_argument('--timeout', type=int, default=50)
  parser.add_argument('--gpu_id', type=int, default=0)
  parser.add_argument('--n_episodes', type=int, default=1001)
  parser.add_argument('--plotting_frequency', type=int, default=100)
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  parser.add_argument('--plot_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_plots')
  parser.add_argument('--use_weighted_classifiers', action='store_true', default=False)
  parser.add_argument('--only_reweigh_negative_examples', action='store_true', default=False)
  parser.add_argument('--always_update', action='store_true', default=False)
  parser.add_argument('--use_gvf_as_initiation_classifier', action='store_true', default=False)
  parser.add_argument('--optimistic_threshold', type=float, default=0.5)
  parser.add_argument('--uncertainty_type', type=str, default='none')
  parser.add_argument('--n_classifier_training_trajectories', type=int, default=10)
  parser.add_argument('--n_classifier_training_epochs', type=int, default=1)
  parser.add_argument('--randomize_direction', action='store_true', default=False)
  parser.add_argument('--level_name', type=str, default='open-grid')
  args = parser.parse_args()

  assert args.level_name in ('open-grid', 'four-rooms')

  g_log_dir = os.path.join(args.log_dir, args.experiment_name, args.sub_dir)
  g_plot_dir = os.path.join(args.plot_dir, args.experiment_name, args.sub_dir)

  utils.create_log_dir(args.log_dir)
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name))
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name, args.sub_dir))
  utils.create_log_dir(g_log_dir)

  utils.create_log_dir(args.plot_dir)
  utils.create_log_dir(os.path.join(args.plot_dir, args.experiment_name))
  utils.create_log_dir(os.path.join(args.plot_dir, args.experiment_name, args.sub_dir))
  utils.create_log_dir(g_plot_dir)

  utils.set_random_seed(args.seed)

  # Create a monte env
  environment = environment_builder(
    level_name="MiniGrid-FourRooms-v0" if args.level_name == 'four-rooms' else 'MiniGrid-Empty-8x8-v0',
    max_steps=1000
  )
  obs0, info0 = environment.reset()

  # s -> o -> measured
  measured_initiations = defaultdict(functools.partial(defaultdict, list))
  ground_truth_initiations = defaultdict(functools.partial(defaultdict, list))

  # Create subgoals and options for MR
  agent_over_options = AgentOverOptions(
    environment,
    n_input_channels=1,
    n_goal_channels=1,
    gestation_period=args.gestation_period,
    timeout=args.timeout, gpu=args.gpu_id,
    image_dim=obs0.shape[-1],
    rams=get_subgoals(environment),
    use_weighted_classifiers=args.use_weighted_classifiers,
    only_reweigh_negative_examples=args.only_reweigh_negative_examples,
    use_gvf_as_initiation_classifier=args.use_gvf_as_initiation_classifier,
    optimistic_threshold=args.optimistic_threshold,
    uncertainty_type=args.uncertainty_type,
    n_classifier_training_trajectories=args.n_classifier_training_trajectories,
    n_classifier_training_epochs=args.n_classifier_training_epochs
  )
  options = agent_over_options.options

  for episode in range(args.n_episodes):

    print("*" * 80)
    print("Episode ", episode)
    print("*" * 80)

    n_gvf_updates = 0

    for state in get_valid_states(environment):

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
        if measured_init or args.always_update:
          n_gvf_updates += len(traj)
          option.update_option_after_rollout(traj, reached)

    # Off-policy things: minibatch updates on GVF and all initiation classifiers 
    if args.use_weighted_classifiers or args.use_gvf_as_initiation_classifier:
      n_gvf_updates = min(max(n_gvf_updates, 500), 5000)  # TODO: Hack to figure out its importance
      print(f'About to perform {n_gvf_updates} GVF updates.')
      agent_over_options.update_initiation_gvf(episode_duration=n_gvf_updates)
    
    # Updating the GVF could have changed the init sets, so re-fit all the clfs
    if not args.use_gvf_as_initiation_classifier:
      print('Updating option initiation classifiers.')
      for option in options:
        option.update_option_initiation_classifiers()
    
    if episode % args.plotting_frequency == 0:
      save_data_tables(episode, ground_truth_initiations, measured_initiations)
      visualize_initiation_table(ground_truth_initiations, options, g_plot_dir, 'gt', episode)
      visualize_initiation_table(measured_initiations, options, g_plot_dir, 'measured', episode)
      if args.use_gvf_as_initiation_classifier:
        [visualize_gc_initiation_learner(
           agent_over_options.initiation_gvf,
           agent_over_options.initiation_gvf.initiation_replay_buffer,
           option.subgoal_obs, option.subgoal_info,
           episode, g_plot_dir, args.experiment_name, args.seed
         ) for option in options]
      else:
        [visualize_initiation_classifier(
          option.initiation_classifier,
          agent_over_options.initiation_gvf,
          agent_over_options.initiation_gvf.initiation_replay_buffer,
          option.subgoal_obs, option.subgoal_info,
          str(option), episode, g_plot_dir,
          args.experiment_name, args.seed) for option in options]
