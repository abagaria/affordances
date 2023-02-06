"""Train a rainbow agent and learn GVF Init w/o impacting decisions in any way."""

import ipdb
import  affordances.utils.plotting as plotting_utils

from affordances.agent.rainbow.rainbow import Rainbow
from affordances.domains.minigrid import environment_builder
from affordances.init_learners.lspi_init_learner import LSPIInitLearner


SEED = 0
EXPERIMENT_NAME = 'AMM_GVF_DEV'


def create_agent(n_actions, env_steps=50_000):
  kwargs = dict(
    n_atoms=51, v_max=10., v_min=-10.,
    noisy_net_sigma=0.5, lr=6.25e-5, n_steps=3,
    betasteps=env_steps // 4,
    replay_start_size=1024, replay_buffer_size=int(5e5),
    gpu=0, n_obs_channels=3, use_custom_batch_states=False
  )
  return Rainbow(n_actions, **kwargs)


def train(agent, init_learner, env, n_episodes=500):
  rewards = []
  for episode in range(n_episodes):
    obs0, info0 = env.reset()
    trajectory, reward = run_episode(agent, init_learner, env, obs0)

    print(f"Episode {episode} S0 {info0['player_pos']} Return {reward} Steps {len(trajectory)}")

    rewards.append(reward)
    agent.experience_replay(trajectory)
    
    plotting_utils.visualize_value_func(
      init_learner,
      None,
      init_learner.replay,
      episode, EXPERIMENT_NAME, SEED)

  return rewards


def run_episode(agent, init_learner, env, obs):
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
    
    # Add transition to the LSPI replay buffer
    init_learner.add_transition(obs, action, reward, next_obs, info)
    init_learner.update()
    
    obs = next_obs

  return trajectory, total_reward


def create_init_learner(env) -> LSPIInitLearner:
  return LSPIInitLearner(env.action_space.n)


  # plotting_utils.visualize_value_func(lspi_agent, env, lspi_replay, params,
  #   episode=-1, experiment_name=EXPERIMENT_NAME, seed=SEED) 


if __name__ == '__main__':
  environment = environment_builder()
  init_learner = create_init_learner(environment)
  rainbow_agent = create_agent(environment.action_space.n)
  reward_list = train(rainbow_agent, init_learner, environment)
