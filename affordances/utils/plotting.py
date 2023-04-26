import ipdb
import collections
import numpy as np
import matplotlib.pyplot as plt
from affordances.utils import utils
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer
from affordances.init_learners.gvf.init_gvf import InitiationGVF


def parse_replay(replay):
  state_dict = {}
  memory = replay.memory.data if isinstance(replay, PrioritizedReplayBuffer) else replay.memory
  for transition in memory:
    transition = transition[-1]  # n-step to single transition
    pos = transition['extra_info']['player_x'], transition['extra_info']['player_y']
    state_dict[pos] = transition['next_state']
  return state_dict


def visualize_value_func(agent, params, replay, episode, experiment_name, seed):
  # Each state has shape (3, 84, 84)
  states, x_positions, y_positions = parse_replay(replay)

  values = []
  chunk_size = min(len(states), 10_000)
  num_chunks = len(states) // chunk_size

  state_chunks = np.array_split(states, num_chunks)
  x_chunks = np.array_split(x_positions, num_chunks)
  y_chunks = np.array_split(y_positions, num_chunks)

  value_dict = collections.defaultdict(list)

  for state_chunk, x_chunk, y_chunk in zip(state_chunks, x_chunks, y_chunks):
    value_chunk = agent.get_values(params, state_chunk)
    value_list = np.asarray(value_chunk).tolist()
    values.extend(value_list)
    for x, y, value in zip(x_chunk, y_chunk, value_list):
      value_dict[(x, y)].append(value)

  x = [pos[0] for pos in value_dict]
  y = [pos[1] for pos in value_dict]
  v = [np.max(value_dict[pos]) for pos in value_dict]
  e = [np.std(value_dict[pos]) for pos in value_dict]

  plt.figure(figsize=(20,12))
  plt.subplot(121)
  plt.scatter(x, y, c=v, s=1000, cmap=plt.cm.Reds, vmin=0, vmax=1)
  plt.colorbar()
  plt.title('Value max')
  plt.subplot(122)
  plt.scatter(x, y, c=e, s=1000, cmap=plt.cm.Reds, vmin=0, vmax=1)
  plt.colorbar()
  plt.title('Value Std Dev')
  plt.savefig(f'plots/{experiment_name}/{seed}/vf_{episode}.png')
  plt.close()


def visualize_initiation_set(option, replay, episode, experiment_name, seed):
  states, x_positions, y_positions = parse_replay(replay)

  values = []
  optimistic_values = []

  chunk_size = min(len(states), 10_000)
  num_chunks = len(states) // chunk_size

  state_chunks = np.array_split(states, num_chunks)
  x_chunks = np.array_split(x_positions, num_chunks)
  y_chunks = np.array_split(y_positions, num_chunks)

  value_dict = collections.defaultdict(list)
  optimistic_value_dict = collections.defaultdict(list)

  for state_chunk, x_chunk, y_chunk in zip(state_chunks, x_chunks, y_chunks):
    value_chunk = option.initiation_learner.pessimistic_predict(state_chunk)
    opt_value_chunk = option.initiation_learner.optimistic_predict(state_chunk)
    value_list = value_chunk.tolist()
    opt_value_list = opt_value_chunk.tolist()

    values.extend(value_list)
    optimistic_values.extend(opt_value_list)

    for x, y, value, opt_value in zip(
      x_chunk,
      y_chunk,
      value_list,
      opt_value_list
    ):
      value_dict[(x, y)].append(value)
      optimistic_value_dict[(x, y)].append(opt_value)

  x = [pos[0] for pos in value_dict]
  y = [pos[1] for pos in value_dict]
  pv = [np.max(value_dict[pos]) for pos in value_dict]
  ov = [np.max(optimistic_value_dict[pos]) for pos in optimistic_value_dict]

  plt.figure(figsize=(20, 12))
  plt.subplot(121)
  plt.scatter(x, y, c=pv, s=1000, cmap=plt.cm.Reds)
  plt.colorbar()
  plt.title(f'Max Pessimistic Init for {option}')
  plt.subplot(122)
  plt.scatter(x, y, c=ov, s=1000, cmap=plt.cm.Reds)
  plt.colorbar()
  plt.title(f'Max Optimistic Init for {option}')

  plt.savefig(f'plots/{experiment_name}/{seed}/{option}_init_{episode}.png')
  plt.close()
  
  
def visualize_gc_initiation_learner(
    initiation_learner, replay, goal, goal_info,
    episode, plot_base_dir, seed,
    save_fig=True
  ):
  
  state_dict = parse_replay(replay)
  value_dict = {}
  for pos in state_dict:
    nsg = state_dict[pos]
    obs = np.asarray(nsg._frames[:-1]).squeeze(1)[None, ...]
    assert obs.shape == (1, 4, 84, 84), obs.shape
    value = initiation_learner.get_values(obs, goal._frames[-1])
    value_dict[pos] = value.item()
  
  x = [pos[0] for pos in value_dict]
  y = [pos[1] for pos in value_dict]
  v = [value_dict[pos] for pos in value_dict]
  plt.scatter(x, y, c=v)
  plt.colorbar()
  gpos = goal_info["player_pos"]
  plt.title(f'Goal pos: {gpos}')
  if save_fig:
    plt.savefig(f'{plot_base_dir}/init_vf_{gpos}_{seed}_{episode}.png')
    plt.close()
  return value_dict


def offline_visuzlize_gc_initiation_learner(
    path_to_init_learner,
    n_actions,
    goal, goal_info,
    episode, plot_base_dir, experiment_name, seed
  ):
  """Visualize the init learner when you just have access to the saved model."""
  assert '.pth' in path_to_init_learner
  init_learner = InitiationGVF(
    target_policy=None,
    n_actions=n_actions,
    n_input_channels=1,
  )
  init_learner.load(path_to_init_learner)
  visualize_gc_initiation_learner(
    init_learner, init_learner.initiation_replay_buffer,
    goal, goal_info, episode, plot_base_dir, experiment_name, seed
  )


def visualize_initiation_classifier(
  initiation_classifier, init_gvf, replay, goal, goal_info,
  option_name, episode, plot_base_dir, experiment_name, seed
):
  state_dict = parse_replay(replay)
  value_dict = {}
  
  for pos in state_dict:
    nsg = state_dict[pos]
    # obs = nsg[0][None, ...][None, ...]
    # assert obs.shape == (1, 1, 84, 84), obs.shape  # [B, C, H, W]
    value = initiation_classifier.optimistic_predict([nsg])
    value_dict[pos] = value.item()
  
  x = [pos[0] for pos in value_dict]
  y = [pos[1] for pos in value_dict]
  v = [value_dict[pos] for pos in value_dict]

  plt.figure(figsize=(20, 12))
  plt.subplot(2, 2, 1)
  plt.scatter(x, y, c=v)
  plt.colorbar()
  pos = goal_info["player_pos"]
  plt.title(f'Classifier: Goal pos: {pos}')

  # Save the lims for plotting egs later
  xlim = min(x), max(x)
  ylim = min(y), max(x)

  # Plot the initiation GVF
  plt.subplot(2, 2, 2)
  visualize_gc_initiation_learner(
    init_gvf, replay, goal, goal_info, episode,
    plot_base_dir, seed, save_fig=False)

  # Plot the training examples
  plt.subplot(2, 2, 3)
  positives = utils.flatten(initiation_classifier.positive_examples)
  negatives = utils.flatten(initiation_classifier.negative_examples)
  
  x_positives = [eg[1]['player_x'] for eg in positives]
  y_positives = [eg[1]['player_y'] for eg in positives]
  x_negatives = [eg[1]['player_x'] for eg in negatives]
  y_negatives = [eg[1]['player_y'] for eg in negatives]

  def eg2val(eg, value_dict):
    if eg[1]['player_pos'] in value_dict:
      return value_dict[eg[1]['player_pos']]
    obs = np.asarray(eg[0]._frames).squeeze(1)[None, ...]
    return init_gvf.get_values(obs, goal._frames[-1]).item()

  weights_positives = [eg2val(eg, value_dict) for eg in positives]
  weights_negatives = [(1 - eg2val(eg, value_dict)) for eg in negatives]
  
  plt.scatter(x_positives, y_positives, c=weights_positives)
  # plt.xlim(xlim)
  # plt.ylim(ylim)
  plt.colorbar()
  plt.title('Clf Positive Examples')

  plt.subplot(2, 2, 4)
  plt.scatter(x_negatives, y_negatives, c=weights_negatives)
  # plt.xlim(xlim)
  # plt.ylim(ylim)
  plt.colorbar()
  plt.title('Clf Negative Examples')

  filename = f'{option_name}_init_vf_{pos}_{seed}_{episode}.png'
  plt.savefig(f'{plot_base_dir}/{filename}')
  plt.close()


def visualize_info_trajectory(traj, fname):
  infos = [trans[1] for trans in traj]
  x = [info['player_x'] for info in infos]
  y = [info['player_y'] for info in infos]
  plt.scatter(x, y)
  plt.savefig(fname)
  plt.close()


def visualize_obs_trajectory(traj, fname):
  lz_frames = [trans[0] for trans in traj]
  for i, stack in enumerate(lz_frames):
    utils.show_frame_stack(stack, f'{i}_{fname}')
