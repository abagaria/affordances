import os
import argparse

from affordances.utils import utils
from affordances.agent.td3.td3 import TD3
from affordances.domains.robot_env_wrapper import make_robosuite_env


def create_agent(
  action_space, 
  obs_space, 
  gpu,
  lr,
  sigma
):
    kwargs = dict(
      obs_space=obs_space, 
      replay_start_size=1024, 
      gpu=gpu,
      sigma=sigma, 
      lr=lr,
      batch_size=32
    )
    return TD3(action_space, **kwargs)


def train(agent: TD3, env, n_episodes):
  episodic_rewards = []
  n_steps = 0
  for episode in range(n_episodes):
    done = False
    episode_reward = 0.
    obs = env.reset_to(None)
    while not done:
      action = agent.act(obs)
      next_obs, reward, done, info = env.step(action)
      agent.step(obs, action, reward, next_obs, done, info['needs_reset'])

      n_steps += 1
      obs = next_obs
      episode_reward += reward
    
    episodic_rewards.append(episode_reward)
    print(f'Episode {episode} Reward {episode_reward}')

    if episode > 0 and episode % 10 == 0:
      utils.safe_zip_write(
        os.path.join(g_log_dir, f'log_seed{args.seed}.pkl'),
        dict(
            rewards=episodic_rewards,
            current_episode=episode,
            current_step_count=n_steps
          )
      )
  return episodic_rewards


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--sub_dir', type=str, default='', help='sub dir for sweeps')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--environment_name', type=str, default='MiniGrid-Empty-8x8-v0')
  parser.add_argument('--n_episodes', type=int, default=5000)
  parser.add_argument('--lr', type=float, default=6.25e-5)
  parser.add_argument('--sigma', type=float, default=0.5)
  parser.add_argument('--bonus_scale', type=float, default=1e-3)
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  args = parser.parse_args()

  g_log_dir = os.path.join(args.log_dir, args.experiment_name, args.sub_dir)

  utils.create_log_dir(args.log_dir)
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name))
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name, args.sub_dir))
  utils.create_log_dir(g_log_dir)

  utils.set_random_seed(args.seed)

  env = make_robosuite_env(args.environment_name, render=False)
  td3_agent = create_agent(
    env.action_space, 
    env.observation_space,
    gpu=args.gpu,
    lr=args.lr,
    sigma=args.sigma
  )

  returns = train(td3_agent, env, args.n_episodes)
