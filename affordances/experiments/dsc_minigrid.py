import os
import random
import argparse
import numpy as np

from affordances.utils import utils
from affordances.agent.dsc.dsc import DSCAgent
from affordances.agent.dsc.option import Option
from affordances.utils import plotting as plotting_utils
from affordances.domains.minigrid import environment_builder, determine_goal_pos
from affordances.goal_attainment.attainment_classifier import DiscreteInfoAttainmentClassifier
from affordances.goal_attainment.attainment_classifier import MultiKeyDiscreteInfoAttainmentClassifier


def create_goal_achievement_classifier(env_name):
  # TODO(ab): Need some way to automate this
  if 'doorkey' in env_name.lower():
    print('[DSC-Minigrid] Using MultiKey GoalAchievement Function')
    return MultiKeyDiscreteInfoAttainmentClassifier(
    key_names=['player_pos', 'is_door_open', 'has_key'])
  print('[DSC-Minigrid] Using SingleKey GoalAchievement Function')
  return DiscreteInfoAttainmentClassifier(key_name='player_pos')


def create_agent(env, s0, i0, goal_info):
  goal_attainment_classifier = create_goal_achievement_classifier(args.environment_name)
  task_goal_classifier = lambda info: goal_attainment_classifier(info, goal_info)
  start_state_classifier = lambda info: goal_attainment_classifier(info, i0)

  return DSCAgent(env, s0, i0, start_state_classifier, task_goal_classifier,
    goal_attainment_classifier,
    args.gestation_period, args.timeout,
    args.init_gvf_type,
    args.init_classifier_type,
    goal_info, args.gpu_id, n_input_channels=1,
    maintain_init_replay=args.plot_initiation_function,
    epsilon_decay_steps=args.epsilon_decay_steps,
    exploration_bonus_scale=args.exploration_bonus_scale,
    use_her_for_policy_evaluation=args.use_her_for_policy_evaluation,
    n_actions=args.n_actions
  )


def train(agent: DSCAgent, env, start_episode, n_episodes):
  rewards = []
  for episode in range(start_episode, start_episode + n_episodes):
    obs0, info0 = env.reset()
    assert not info0['needs_reset'], info0
    state, info, episode_rewards = agent.dsc_rollout(obs0, info0)
    undiscounted_return = sum(episode_rewards)
    rewards.append(undiscounted_return)
    print(100 * '-')
    print(f'Episode: {episode}',
      f"InitPos': {info0['player_pos']}",
      f"GoalPos: {goal_info_dict['player_pos']}",
      f"FinalPos: {info['player_pos']}",
      f'Reward: {undiscounted_return}')
    print(100 * '-')
  return rewards


def test(agent: DSCAgent, env, n_episodes):
  rewards = []
  for episode in range(n_episodes):
    env.reset()
    obs0, info0 = env.reset_to(agent.start_state_info['player_pos'])
    assert not info0['needs_reset'], info0
    state, info, episode_rewards = agent.dsc_rollout(obs0, info0)
    undiscounted_return = sum(episode_rewards)
    rewards.append(undiscounted_return)
    print(100 * '-')
    print(f'[Test] Episode: {episode}',
      f"InitPos': {info0['player_pos']}",
      f"GoalPos: {goal_info_dict['player_pos']}",
      f"FinalPos: {info['player_pos']}",
      f'Reward: {undiscounted_return}')
    print(100 * '-')
  return rewards


def log(agent: DSCAgent, returns_so_far: list, episode: int):
  """Log DSC progress: includes learning curve, plotting, checkpointing."""

  option_logs = dict()

  for option in agent.chain:
    option_logs[str(option)] = option.debug_log

  utils.safe_zip_write(
    f'{g_log_dir}/log_seed{args.seed}.pkl',
    dict(
      rewards=returns_so_far,
      current_episode=episode,
      option_logs=option_logs,
      start=start_info,
      goal=goal_info_dict
    )
  )

  if args.checkpoint_init_learners and episode % args.checkpoint_frequency == 0:
    filename = f'{g_log_dir}/init_learner_{episode}.pth'
    agent.initiation_gvf.save(filename)
  
  if episode > 0 and args.plotting_frequency > 0 \
      and episode % args.plotting_frequency == 0:
    for option in agent.mature_options:
      assert isinstance(option, Option)
      goal, goal_info = option.sample_goal(
        *random.choice(
          utils.flatten(option.initiation_classifier.positive_examples)
        )
      )
      plotting_utils.visualize_initiation_classifier(
        option.initiation_classifier,
        init_gvf=agent.initiation_gvf,
        replay=agent.initiation_gvf.initiation_replay_buffer,
        goal=goal,
        goal_info=goal_info,
        option_name=str(option),
        episode=episode,
        plot_base_dir=args.plot_dir,
        experiment_name=args.experiment_name,
        seed=args.seed
      )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--sub_dir', type=str, default='', help='sub dir for sweeps')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--environment_name', type=str, default='MiniGrid-Empty-8x8-v0')
  parser.add_argument('--gestation_period', type=int, default=5)
  parser.add_argument('--timeout', type=int, default=50)
  parser.add_argument('--init_gvf_type', type=str, default='td0')
  parser.add_argument('--init_classifier_type', type=str, default='weighted-binary')
  parser.add_argument('--gpu_id', type=int, default=0)
  parser.add_argument('--n_episodes', type=int, default=5000)
  parser.add_argument('--plot_initiation_function', action='store_true', default=False)
  parser.add_argument('--checkpoint_init_learners', action='store_true', default=False)
  parser.add_argument('--checkpoint_frequency', type=int, default=100)
  parser.add_argument('--plotting_frequency', type=int, default=-1)
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  parser.add_argument('--plot_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_plots')
  parser.add_argument('--exploration_bonus_scale', default=0, type=float)
  parser.add_argument('--epsilon_decay_steps', type=int, default=25_000)
  parser.add_argument('--final_epsilon', type=float, default=0.1)
  parser.add_argument('--use_random_resets', action='store_true', default=False)
  parser.add_argument('--use_her_for_policy_evaluation', action='store_true', default=False)
  parser.add_argument('--n_actions', type=int, help='Specify the number of actions in the env')
  parser.add_argument('--max_env_steps', type=int, default=-1, help='steps/episode')
  args = parser.parse_args()

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

  environment = environment_builder(
    level_name=args.environment_name,
    seed=args.seed,
    exploration_reward_scale=0,
    random_reset=args.use_random_resets,
    max_steps=args.max_env_steps
  )
  
  # Assuming that env.env corresponds to the InfoWrapper
  start_state = environment.official_start_obs
  start_info = environment.official_start_info

  goal_info_dict = dict(player_pos=determine_goal_pos(environment))
  print(environment)
  dsc_agent = create_agent(environment, start_state, start_info, goal_info_dict)

  episodic_returns = []

  # If we use random resets, we do separate train and test rollouts
  if args.use_random_resets:
    current_training_episode = 0
    for iteration in range(args.n_episodes // 10):
      training_returns = train(dsc_agent,
                               environment,
                               start_episode=current_training_episode,
                               n_episodes=10)
      evaluation_returns = test(dsc_agent, environment, 5)
      episodic_returns.append(np.mean(evaluation_returns))
      current_training_episode += 10
      log(dsc_agent, episodic_returns, current_training_episode)
  else:
    episodic_returns = train(dsc_agent, environment, 0, args.n_episodes)

  # Plotting at the end so that even if we OOM, it doesn't matter
  plotting_utils.visualize_gc_initiation_learner(
    dsc_agent.initiation_gvf,
    dsc_agent.initiation_gvf.initiation_replay_buffer,
    *dsc_agent.get_subgoal_for_global_option(start_state),
    args.n_episodes,
    args.plot_dir,
    args.experiment_name,
    args.seed,
    save_fig=True
  )
