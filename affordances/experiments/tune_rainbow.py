import pickle
import argparse

from affordances.utils import utils
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.domains.minigrid import environment_builder


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
      gpu=gpu, n_obs_channels=n_input_channels,
      use_custom_batch_states=False,
      epsilon=0.1
    )
    return Rainbow(n_actions, **kwargs)


def train(agent: Rainbow, env, n_episodes):
  episodic_rewards = []
  for episode in range(n_episodes):
    done = False
    episode_reward = 0.
    obs, info = env.reset()
    while not done:
      action = agent.act(obs)
      next_obs, reward, done, info = env.step(action)
      agent.step(obs, action, reward, next_obs, done, info['needs_reset'])

      obs = next_obs
      episode_reward += reward
    
    episodic_rewards.append(episode_reward)
    print(f'Episode {episode} Reward {episode_reward}')

    if episode > 0 and episode % 10 == 0:
      with open(f'logs/{args.experiment_name}/{args.seed}/log.pkl', 'wb+') as f:
        pickle.dump(
          dict(
            rewards=episodic_rewards,
            current_episode=episode,
          ), f)
  return episodic_rewards


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--environment_name', type=str, default='MiniGrid-Empty-8x8-v0')
  parser.add_argument('--n_episodes', type=int, default=5000)
  parser.add_argument('--lr', type=float, default=6.25e-5)
  parser.add_argument('--sigma', type=float, default=0.5)
  args = parser.parse_args()

  utils.create_log_dir(f'logs/{args.experiment_name}')
  utils.create_log_dir(f'logs/{args.experiment_name}/{args.seed}')

  environment = environment_builder(args.environment_name)
  rainbow_agent = create_agent(
    environment.action_space.n,
    gpu=args.gpu,
    n_input_channels=1,
    lr=args.lr,
    sigma=args.sigma
  )

  returns = train(rainbow_agent, environment, args.n_episodes)
