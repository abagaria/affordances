import os
import sys 
import copy
import argparse

import torch
import numpy as np 
from tqdm import tqdm

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
      only_reweigh_negative_examples=args.only_reweigh_negatives,
      gestation_period=args.gestation
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
    uncertainty_type=args.uncertainty,
    bonus_scale=args.bonus_scale
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

def kernel_count(states, memory, eps=1e-3, c=1e-3, k=2):
  # TODO: moving average
  # TODO: cluster
  r_t = np.zeros(len(states))
  for i, state in enumerate(states):
    dists = sorted([np.linalg.norm(state - m) for m in memory])
    dists = dists[:k]
    dists = np.array(dists)**2
    if sum(dists) > 0:
      dists = dists / np.mean(dists)
    Ks = eps / (dists + eps)
    denom = np.sqrt(np.sum(Ks)) + c
    r_t[i] = 1 / denom
    print(dists)
  return r_t

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
  parser.add_argument('--bonus_scale', type=float, default=0.0)
  parser.add_argument('--gestation', type=int, default=1)
  parser.add_argument('--only_reweigh_negatives', type=utils.boolify, default=False)
  parser.add_argument('--optimal_ik', type=utils.boolify, default=False)
  parser.add_argument('--segment', type=utils.boolify, default=False)
  parser.add_argument('--render', type=utils.boolify, default=False)
  parser.add_argument('--vis_init_set', type=utils.boolify, default=False)
  parser.add_argument('--eval_accuracy', type=utils.boolify, default=False)
  args = parser.parse_args()
  print(args)

  # maybe exit for nonsense arg combos from onager 
  if args.uncertainty == 'bonus':
    assert args.init_learner == 'gvf' or args.init_learner == 'weighted-binary'

  if args.uncertainty == 'none':
    assert args.bonus_scale == 0
  else:
    assert args.bonus_scale > 0

  if args.init_learner == 'random':
    assert args.bonus_scale == 0
    assert args.uncertainty == 'none'
    assert args.gestation == 1 
    assert args.only_reweigh_negatives == False
    assert args.sampler == 'sum'

  if args.only_reweigh_negatives:
    assert args.init_learner == 'weighted-binary'

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
  EVAL_FREQ = len(grasp_qpos)
  episodic_rewards = []
  episodic_success = []
  states_attempted = []
  accuracy_log = []
  
  n_steps = 0
  for episode in range(args.n_episodes + 1):
    sys.stdout.flush()

    # score
    grasp_scores = init_learner.score(grasp_state_vectors)

    # maybe bonus
    if args.uncertainty == "count_qpos":
      bonus = count_uncertainty(qpos_counts)    
    elif args.uncertainty == "count_grasp":
      bonus = count_uncertainty(grasp_counts)
      bonus = np.repeat(bonus, n_qpos_per_grasp)
    elif args.uncertainty == "kernel":
      bonus = kernel_count(grasp_qpos, states_attempted)
    else:
      bonus = 0.
    grasp_scores += args.bonus_scale * bonus

    # maybe vis 
    if args.init_learner != "random" and args.vis_init_set and episode % 500 == 0:
      mask = grasp_scores > init_learner.optimistic_threshold
      env.reset()
      env.render_states(grasp_qpos[mask], ep_num=episode, fpath=g_log_dir)

    # maybe eval accuracy 
    if args.init_learner != "random" and args.eval_accuracy and episode % EVAL_FREQ == 0:
      accuracy = []
      num_success = 0
      num_acc = 0 
      size = sum(grasp_scores > init_learner.optimistic_threshold)
      for i, state in tqdm(enumerate(grasp_qpos)):
        obs = env.reset_to_joint_state(state)
        done = False
        prediction = grasp_scores[i] > init_learner.optimistic_threshold
        while not done:
          action = agent.act(obs)
          next_obs, reward, done, info = env.step(action)
          obs = next_obs
          success = info['success']
          if success:
            assert done
        accuracy.append( (success, prediction) )
        num_success += success
        num_acc += (success == prediction)
      num_acc = num_acc / len(grasp_scores)
      accuracy_log.append(accuracy)
      print(f'Evaluation: Accuracy {num_acc}, Pred Size {size}/{len(grasp_scores)}, True Size {num_success}/{len(grasp_scores)}')

    # reset 
    done = False
    episode_reward = 0.
    selected_idx = sample_func(grasp_scores)
    selected_qpos = grasp_qpos[selected_idx]
    selected_grasp_idx = selected_idx // n_qpos_per_grasp
    obs = env.reset_to_joint_state(selected_qpos)
    states_attempted.append(selected_qpos)
    
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

    if episode % 10 == 0:
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
            qpos_success=qpos_success,
            accuracy=accuracy_log
          )
      )

  # save 
  init_learner_fname = os.path.join(g_log_dir, 'init.pth')
  if args.init_learner != "random":
    init_learner.save(init_learner_fname)
  agent.agent.save(g_log_dir)


