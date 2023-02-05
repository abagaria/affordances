import argparse
from pfrl.replay_buffers import ReplayBuffer
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.domains.minigrid import environment_builder
from affordances.agent.td.td_policy_eval import TDPolicyEvaluator
from affordances.utils import utils

import affordances.utils.plotting as plotting_utils


def create_agent(n_actions, env_steps=50_000):
  kwargs = dict(
    n_atoms=51, v_max=10., v_min=-10.,
    noisy_net_sigma=0.5, lr=6.25e-5, n_steps=3,
    betasteps=env_steps // 4,
    replay_start_size=1024, replay_buffer_size=int(5e5),
    gpu=0, n_obs_channels=3, use_custom_batch_states=False
  )
  return Rainbow(n_actions, **kwargs)


def run_episode(agent, env, obs):
  info = {}
  done = False
  total_reward = 0.
  trajectory = []

  while not done:
    action = agent.act(obs)
    next_obs, reward, done, info = env.step(action)

    total_reward += reward
    
    transition = obs, action, reward, next_obs, info['terminated'], info['needs_reset']
    trajectory.append(transition)

    # Add transition to the GVF replay buffer
    initiation_replay_buffer.append(obs, action, reward, next_obs,
      is_state_terminal=info['terminated'], extra_info=info)

    obs = next_obs

  return trajectory, total_reward


def train(agent: Rainbow, init_learner, env, n_episodes=500):
  rewards = []
  for episode in range(n_episodes):
    obs0, info0 = env.reset()
    trajectory, reward = run_episode(agent, env, obs0)

    # Fit the initiation set classifier
    if len(initiation_replay_buffer) > init_learner._batch_size:
      target_policy = agent.agent.batch_act
      init_learner.train(target_policy)

    print(f"Episode {episode} S0 {info0['player_pos']} Return {reward}")

    rewards.append(reward)
    agent.experience_replay(trajectory)

    plotting_utils.visualize_value_func(initiation_agent, None, initiation_replay_buffer,
      episode=episode, experiment_name=args.experiment_name, seed=args.seed)

  return rewards


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--use_prioritized_sampling', action='store_true', default=False)
  args = parser.parse_args()

  utils.create_log_dir(f'plots/{args.experiment_name}')
  utils.create_log_dir(f'plots/{args.experiment_name}/{args.seed}')

  environment = environment_builder()
  rainbow_agent = create_agent(environment.action_space.n)
  
  initiation_replay_buffer = PrioritizedReplayBuffer(capacity=int(1e5)) if args.use_prioritized_sampling else ReplayBuffer(capacity=int(1e5))
  initiation_agent = TDPolicyEvaluator(initiation_replay_buffer, environment.action_space.n)
  
  reward_list = train(rainbow_agent, initiation_agent, environment)
