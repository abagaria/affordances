"""Train a rainbow agent and learn GVF Init w/o impacting decisions in any way."""

import ipdb
import  affordances.utils.plotting as plotting_utils

from affordances.agent.lspi.lspi import LSPI
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.domains.minigrid import environment_builder
from pfrl.replay_buffers import ReplayBuffer


EXPERIMENT_NAME = 'gvf_dev'
SEED = 0


lspi_replay = ReplayBuffer(capacity=int(1e5))


def create_agent(n_actions, env_steps=50_000):
  kwargs = dict(
    n_atoms=51, v_max=10., v_min=-10.,
    noisy_net_sigma=0.5, lr=6.25e-5, n_steps=3,
    betasteps=env_steps // 4,
    replay_start_size=1024, replay_buffer_size=int(1e6),
    gpu=0, n_obs_channels=3, use_custom_batch_states=False
  )
  return Rainbow(n_actions, **kwargs)


def train(agent, env, n_episodes=500):
  rewards = []
  for episode in range(n_episodes):
    obs0, info0 = env.reset()
    trajectory, reward = run_episode(agent, env, obs0)

    # Fit the initiation set classifier
    if episode > 1:
      fit_initiation_set(lspi_replay, env, episode)

    print(f"Episode {episode} S0 {info0['player_pos']} Return {reward}")

    rewards.append(reward)
    agent.experience_replay(trajectory)

  return rewards


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
    
    # Add transition to the LSPI replay buffer
    lspi_replay.append(obs, action, reward, next_obs,
      is_state_terminal=info['terminated'], extra_info=info)
    
    obs = next_obs

  return trajectory, total_reward


def create_init_learner(transitions, env):
  n_obs_features = 84 * 84 * 3  # number of features (pixels) in raw obs
  return LSPI(*transitions, n_actions=env.action_space.n,
    n_state_features=n_obs_features, extraction_method='random')


def fit_initiation_set(replay, env, episode=-1, batch_size=1024):

  # Sample and prepare transitions
  n_samples = min(len(replay), batch_size)

  sampled_n_step_transitions = replay.sample(n_samples)

  # Prepare transitions for LSPI
  sampled_transitions = [n_step_transitions[-1] for n_step_transitions in sampled_n_step_transitions]
  parsed_transitions = unpack_transitions(sampled_transitions)
  
  # Fit a value function \hat{V^*}
  lspi_agent = create_init_learner(parsed_transitions, env)
  params = lspi_agent()

  values = lspi_agent.get_values(params)
  print(f'Values min: {values.min()} max: {values.max()} mean: {values.mean()}')

  plotting_utils.visualize_value_func(lspi_agent, params, lspi_replay,
    episode=episode, experiment_name=EXPERIMENT_NAME, seed=SEED) 


def unpack_transitions(transitions):
  states = []
  actions = []
  rewards = []
  next_states = []
  dones = []
  for transition in transitions:
    states.append(transition['state'].transpose((1, 2, 0)))
    actions.append(transition['action'])
    rewards.append(transition['reward'])
    next_states.append(transition['next_state'].transpose((1, 2, 0)))
    dones.append(transition['is_state_terminal'])
  return states, actions, rewards, next_states, dones


if __name__ == '__main__':
  environment = environment_builder()
  rainbow_agent = create_agent(environment.action_space.n)
  reward_list = train(rainbow_agent, environment)
