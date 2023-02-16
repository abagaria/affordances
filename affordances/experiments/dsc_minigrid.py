import os
import argparse

from affordances.utils import utils
from affordances.agent.dsc.dsc import DSCAgent
from affordances.utils import plotting as plotting_utils
from affordances.domains.minigrid import environment_builder, determine_goal_pos
from affordances.goal_attainment.attainment_classifier import DiscreteInfoAttainmentClassifier


def create_agent(env, s0, i0, goal_info):
  goal_attainment_classifier = DiscreteInfoAttainmentClassifier(
    key_name='player_pos')
  task_goal_classifier = lambda info: goal_attainment_classifier(info, goal_info)

  return DSCAgent(env, s0, i0, task_goal_classifier, goal_attainment_classifier,
    args.gestation_period, args.timeout, args.init_learner_type,
    goal_info, args.gpu_id, n_input_channels=1,
    maintain_init_replay=args.plot_initiation_function)


def train(agent: DSCAgent, env, n_episodes):
  rewards = []
  for episode in range(n_episodes):
    obs0, info0 = env.reset()
    state, info, episode_rewards = agent.dsc_rollout(obs0, info0)
    undiscounted_return = sum(episode_rewards)
    rewards.append(undiscounted_return)
    print(f'Episode: {episode}',
      f"InitPos': {info0['player_pos']}",
      f"FinalPos: {info['player_pos']}",
      f'Reward: {undiscounted_return}')
    log(agent, rewards, episode)
  return rewards


def log(agent: DSCAgent, returns_so_far: list, episode: int):
  """Log DSC progress: includes learning curve, plotting, checkpointing."""

  utils.safe_zip_write(
    f'{g_log_dir}/log.pkl',
    dict(
      rewards=returns_so_far,
      current_episode=episode
    )
  )

  if args.checkpoint_init_learners and episode % args.checkpoint_frequency == 0:
    for option in agent.mature_options:
      filename = f'{g_log_dir}/option_{option.option_idx}_init.pth'
      option.initiation_learner.save(filename)

  if args.plot_initiation_function and episode % args.plotting_frequency == 0:
    [plotting_utils.visualize_initiation_set(
      option, agent._init_replay_buffer,
      episode, args.experiment_name, args.seed
    ) for option in agent.mature_options]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--sub_dir', type=str, default='', help='sub dir for sweeps')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--environment_name', type=str, default='MiniGrid-Empty-8x8-v0')
  parser.add_argument('--gestation_period', type=int, default=5)
  parser.add_argument('--timeout', type=int, default=50)
  parser.add_argument('--init_learner_type', type=str, default='binary')
  parser.add_argument('--gpu_id', type=int, default=0)
  parser.add_argument('--n_episodes', type=int, default=5000)
  parser.add_argument('--plot_initiation_function', action='store_true', default=False)
  parser.add_argument('--checkpoint_init_learners', action='store_true', default=False)
  parser.add_argument('--checkpoint_frequency', type=int, default=100)
  parser.add_argument('--plotting_frequency', type=int, default=1)
  args = parser.parse_args()

  g_log_dir = os.path.join('logs', args.experiment_name, args.sub_dir, str(args.seed))
  g_plot_dir = os.path.join('plots', args.experiment_name, args.sub_dir, str(args.seed))

  utils.create_log_dir('logs')
  utils.create_log_dir(os.path.join('logs', args.experiment_name))
  utils.create_log_dir(os.path.join('logs', args.experiment_name, args.sub_dir))
  utils.create_log_dir(g_log_dir)

  utils.create_log_dir('plots')
  utils.create_log_dir(os.path.join('plots', args.experiment_name))
  utils.create_log_dir(os.path.join('plots', args.experiment_name, args.sub_dir))
  utils.create_log_dir(g_plot_dir)

  utils.set_random_seed(args.seed)

  environment = environment_builder(
    level_name=args.environment_name, seed=args.seed)
  start_state, start_info = environment.reset()
  goal_info_dict = dict(player_pos=determine_goal_pos(environment))
  dsc_agent = create_agent(environment, start_state, start_info, goal_info_dict)

  episodic_returns = train(dsc_agent, environment, args.n_episodes)
