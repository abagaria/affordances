import os
import sys 
import argparse

import torch
import numpy as np 

from affordances.utils import utils
from affordances.agent.td3.td3 import TD3
from affordances.domains.cip_env_wrapper import make_robosuite_env

from affordances.init_learners.classification.binary_init_classifier import MlpInitiationClassifier
from affordances.init_learners.classification.random_grasp_baseline import RandomGraspBaseline
from affordances.init_learners.gvf.init_gvf import InitiationGVF

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
      batch_size=256
    )
    return TD3(action_space, **kwargs)

def create_init_learner(args, env, agent):
  if args.init_learner == "random":
    return RandomGraspBaseline()
  elif "binary" in args.init_learner:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = env.observation_space.shape[0]
    optimistic_threshold=0.5,
    pessimistic_threshold=0.75,
    return MlpInitiationClassifier(
      device, 
      optimistic_threshold, 
      pessimistic_threshold, 
      input_dim, 
      maxlen=100, 
      only_reweigh_negative_examples=args.only_reweigh_negatives
    )
  elif args.init_learner == "gvf":
    return create_gvf(args, env, agent)
  else:
    raise ValueError("invalid init_learner")

def create_gvf(args, env, agent):
  return InitiationGVF(
    agent.get_actions, 
    env.action_space.shape[0],
    env.observation_space.shape[0],
    optimistic_threshold=0.5,
    uncertainty_type=args.uncertainty
  )

def create_sample_func(args):
  """ given a list of scores, return index chosen """
  if args.sampler == "max":
    return lambda x: np.argmax(x)
  elif args.sampler == "sum":
    return lambda x: np.random.choice(np.arange(len(x)), p=(x/sum(x)) )
  else:
    raise ValueError("args.sampler is invalid [max, sum]")

def relabel_trajectory(transitions):
  relabeled_trajectory = []
  for state, action, _, next_state, info in transitions:
    reached = info['success']
    reward = float(reached)
    relabeled_trajectory.append((state, action, reward, next_state, reached, info))
    if reached:
      break
  return relabeled_trajectory

def count_uncertainty(counts):
    return 1.0 / np.sqrt(counts + 1.0)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment_name', type=str)
  parser.add_argument('--sub_dir', type=str, default='', help='sub dir for sweeps')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--environment_name', type=str, default='MiniGrid-Empty-8x8-v0')
  parser.add_argument('--n_episodes', type=int, default=500)
  parser.add_argument('--lr', type=float, default=3e-4)
  parser.add_argument('--sigma', type=float, default=0.05)
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  parser.add_argument('--sampler', type=str, default='sum', help='[sum, max]')
  parser.add_argument('--init_learner', type=str, default='binary')
  parser.add_argument('--uncertainty', type=str, default='none', help='[none, bonus, count_qpos, count_grasp]')
  parser.add_argument('--bonus_scale', type=float, default=1.0)
  parser.add_argument('--only_reweigh_negatives', type=utils.boolify, default=False)
  parser.add_argument('--optimal_ik', type=utils.boolify, default=False)
  parser.add_argument('--segment', type=utils.boolify, default=False)
  parser.add_argument('--render', type=utils.boolify, default=False)
  parser.add_argument('--vis_init_set', type=utils.boolify, default=False)
  args = parser.parse_args()
  print(args)

  if args.uncertainty != 'none':
    assert args.init_learner == 'gvf'

  g_log_dir = os.path.join(args.log_dir, args.experiment_name, args.sub_dir)

  utils.create_log_dir(args.log_dir)
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name))
  utils.create_log_dir(os.path.join(args.log_dir, args.experiment_name, args.sub_dir))
  utils.create_log_dir(g_log_dir)

  utils.set_random_seed(args.seed)

   # log args
  run_dict = vars(args)
  run_dict_file = os.path.join(g_log_dir, "config.pkl")
  utils.safe_zip_write(run_dict_file, run_dict)

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

  init_learner = create_init_learner(args, env, agent)
  sample_func = create_sample_func(args)

  # maybe make external gvf
  if args.init_learner == "weighted-binary":
    init_gvf = create_gvf(args, env, agent)
  else:
    init_gvf = None
  
  # load grasps 
  n_qpos_per_grasp = 1 if env.optimal_ik else 5
  grasps = env.load_grasps()
  grasp_state_vectors, grasp_qpos = env.get_states_from_grasps(n=n_qpos_per_grasp)

  # init counts
  qpos_counts = np.zeros(len(grasp_state_vectors))
  qpos_success = np.zeros(len(grasp_state_vectors))
  grasp_counts = np.zeros(len(grasps))
  grasp_success = np.zeros(len(grasps))

  # train
  episodic_rewards = []
  episodic_success = []
  n_steps = 0
  for episode in range(args.n_episodes):
    sys.stdout.flush()
    done = False
    episode_reward = 0.

    # score
    grasp_scores = init_learner.score(grasp_state_vectors)

    # maybe bonus
    if args.uncertainty == "count_qpos":
        bonus = count_uncertainty(qpos_counts)
        grasp_scores += args.bonus_scale * bonus
    elif args.uncertainty == "count_grasp":
        bonus = count_uncertainty(grasp_counts)
        bonus = np.repeat(bonus, n_qpos_per_grasp)
        grasp_scores += args.bonus_scale * bonus

    # maybe vis 
    if args.init_learner != "random" and args.vis_init_set and episode % 500 == 0:
      mask = grasp_scores > init_learner.optimistic_threshold
      env.reset()
      env.render_states(grasp_qpos[mask], ep_num=episode, fpath=g_log_dir)

    # reset 
    selected_idx = sample_func(grasp_scores)
    selected_qpos = grasp_qpos[selected_idx]
    selected_grasp_idx = selected_idx // n_qpos_per_grasp
    obs = env.reset_to_joint_state(selected_qpos)
    
    trajectory = []  # (s, a, r, s', info)
    while not done:
      action = agent.act(obs)
      next_obs, reward, done, info = env.step(action)
      agent.step(obs, action, reward, next_obs, done, info['needs_reset'])
      trajectory.append((obs, action, reward, next_obs, info))

      n_steps += 1
      obs = next_obs
      episode_reward += reward
    
    success = info['success']
    episodic_rewards.append(episode_reward)
    episodic_success.append(success)
    grasp_counts[selected_grasp_idx] += 1
    grasp_success[selected_grasp_idx] += info['success']
    qpos_counts[selected_idx] += 1
    qpos_success[selected_idx] += info['success']
    
    print(f'Episode {episode} Reward {episode_reward} Success {success}')

    # maybe update gvf
    if init_gvf is not None:
      init_gvf.add_trajectory_to_replay(relabel_trajectory(trajectory))
      init_gvf.update()
    
    # update init_learner
    if type(init_learner) is InitiationGVF:
      init_learner.add_trajectory_to_replay(relabel_trajectory(trajectory))
      init_learner.update()
    else:
      init_learner.add_trajectory(trajectory, success)
      init_learner.update(initiation_gvf=init_gvf)

    if episode > 0 and episode % 10 == 0:
      utils.safe_zip_write(
        os.path.join(g_log_dir, f'log_seed{args.seed}.pkl'),
        dict(
            rewards=episodic_rewards,
            success=episodic_success,
            current_episode=episode,
            current_step_count=n_steps,
            classifier_loss=init_learner.classifier.losses if type(init_learner) is MlpInitiationClassifier else [],
            scores=grasp_scores,
            grasp_counts=grasp_counts,
            grasp_success=grasp_success,
            qpos_counts=qpos_counts,
            qpos_success=qpos_success
          )
      )

  # save 
  init_learner_fname = os.path.join(g_log_dir, 'init.pth')
  if args.init_learner != "random":
    init_learner.save(init_learner_fname)
  agent.agent.save(g_log_dir)


