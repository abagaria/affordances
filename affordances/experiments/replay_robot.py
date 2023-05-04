import os
import sys 
import gzip
import pickle
import argparse

import torch
import numpy as np 

from affordances.utils import utils
from affordances.agent.td3.td3 import TD3
from affordances.domains.cip_env_wrapper import make_robosuite_env
from affordances.experiments.robot_td3 import create_agent, create_init_learner, create_gvf, create_sample_func


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--path', type=str)
  args = parser.parse_args()

   # log args
  run_dict_file = os.path.join(args.path, "config.pkl")
  with gzip.open(run_dict_file, 'rb+') as f:
    job_data = pickle.load(f)

  args.environment_name = job_data['environment_name']
  args.render = True
  args.optimal_ik = job_data['optimal_ik']
  args.init_learner = job_data['init_learner']
  args.segment = job_data['segment']
  args.lr = job_data['lr']
  args.sigma = job_data['lr']
  args.sampler = job_data['sampler']
  args.gpu = job_data['gpu']

  # create env, agent, init_learner
  env = make_robosuite_env(
    args.environment_name, 
    deterministic=True, 
    render=args.render, 
    use_qpos_cache=True,
    optimal_ik=args.optimal_ik,
    segment=args.segment
  )
  agent = create_agent(
    env.action_space, 
    env.observation_space,
    gpu=args.gpu if torch.cuda.is_available() else None,
    lr=args.lr,
    sigma=args.sigma
  )
  agent.agent.load(args.path)
  agent.agent.eval_mode()

  init_learner = create_init_learner(args, env, agent)
  init_learner_fname = os.path.join(args.path, 'init.pth')
  init_learner.load(init_learner_fname)

  sample_func = create_sample_func(args)

  # load grasps 
  qpos_per_grasp = 1 if env.optimal_ik else 5
  grasps = env.load_grasps()
  env.set_render(False)
  grasp_state_vectors, grasp_qpos = env.get_states_from_grasps(n=qpos_per_grasp)
  env.set_render(True)
  
  # eval 
  for episode in range(10000):
    sys.stdout.flush()
    done = False
    episode_reward = 0.

    grasp_scores = init_learner.score(grasp_state_vectors)
    selected_idx = sample_func(grasp_scores)
    selected_qpos = grasp_qpos[selected_idx]
    obs = env.reset_to_joint_state(selected_qpos)
    
    trajectory = []  # (s, a, r, s', info)
    while not done:
      action = agent.act(obs)
      next_obs, reward, done, info = env.step(action)
      agent.step(obs, action, reward, next_obs, done, info['needs_reset'])
      trajectory.append((obs, action, reward, next_obs, info))
      obs = next_obs
      episode_reward += reward
    
    success = info['success']
    print(f'Episode {episode} Reward {episode_reward} Success {success}')