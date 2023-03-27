import ipdb
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer
from affordances import utils
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


def visualize_gc_initiation_learner(
    initiation_learner, replay, goal, goal_info,
    episode, plot_base_dir, experiment_name, seed
  ):
  
  state_dict = parse_replay(replay)
  value_dict = {}
  for pos in state_dict:
    nsg = state_dict[pos]
    obs = nsg[0][None, ...]
    assert obs.shape == (1, 84, 84), obs.shape
    value = initiation_learner.get_values([obs], [goal])
    value_dict[pos] = value.item()
  
  x = [pos[0] for pos in value_dict]
  y = [pos[1] for pos in value_dict]
  v = [value_dict[pos] for pos in value_dict]
  plt.scatter(x, y, c=v)
  plt.colorbar()
  plt.title(f'Goal pos: {goal_info["player_pos"]}')
  plt.savefig(f'{plot_base_dir}/{experiment_name}/init_vf_{seed}_{episode}.png')
  plt.close()


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
