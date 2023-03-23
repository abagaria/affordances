import os
import random
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
  start_state_classifier = lambda info: goal_attainment_classifier(info, i0)

  return DSCAgent(env, s0, i0, start_state_classifier, task_goal_classifier,
    goal_attainment_classifier,
    args.gestation_period, args.timeout, args.init_learner_type,
    goal_info, args.gpu_id, n_input_channels=1,
    maintain_init_replay=args.plot_initiation_function,
    epsilon_decay_steps=args.epsilon_decay_steps,
    exploration_bonus_scale=args.exploration_bonus_scale,
    use_her_for_policy_evaluation=args.use_her_for_policy_evaluation
  )


def train(agent: DSCAgent, env, n_episodes):
  rewards = []
  for episode in range(n_episodes):
    obs0, info0 = env.reset()
    state, info, episode_rewards = agent.dsc_rollout(obs0, info0)
    undiscounted_return = sum(episode_rewards)
    rewards.append(undiscounted_return)
    print(80 * '-')
    print(f'Episode: {episode}',
      f"InitPos': {info0['player_pos']}",
      f"FinalPos: {info['player_pos']}",
      f'Reward: {undiscounted_return}')
    print(80 * '-')
    log(agent, rewards, episode)
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
    for option in agent.mature_options:
      filename = f'{g_log_dir}/option_{option._option_idx}_init_{episode}.pth'
      option.initiation_learner.save(filename)
      break
    
    if agent._init_replay_buffer is not None:
      agent._init_replay_buffer.save(f'{g_log_dir}/init_replay_{episode}.pkl')

  # for option in agent.mature_options:
  #   goal, goal_info = random.choice(option.get_potential_goals())
  #   plotting_utils.visualize_gc_initiation_learner(
  #     agent.initiation_learner, agent.initiation_learner.initiation_replay_buffer,
  #     goal, goal_info, episode, 'plots', args.experiment_name, args.seed
  #   )


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
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  parser.add_argument('--exploration_bonus_scale', default=0, type=float)
  parser.add_argument('--epsilon_decay_steps', type=int, default=25_000)
  parser.add_argument('--use_random_resets', action='store_true', default=False)
  parser.add_argument('--use_her_for_policy_evaluation', action='store_true', default=False)
  args = parser.parse_args()

  g_log_dir = os.path.join(args.log_dir, args.experiment_name, args.sub_dir)

  utils.create_log_dir(args.log_dir)
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name))
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name, args.sub_dir))
  utils.create_log_dir(g_log_dir)

  utils.set_random_seed(args.seed)

  environment = environment_builder(
    level_name=args.environment_name, seed=args.seed,
    exploration_reward_scale=0,
    random_reset=args.use_random_resets
  )
  
  start_state, start_info = environment.reset()
  goal_info_dict = dict(player_pos=determine_goal_pos(environment))
  print(environment)
  dsc_agent = create_agent(environment, start_state, start_info, goal_info_dict)

  episodic_returns = train(dsc_agent, environment, args.n_episodes)
