import ipdb
import collections
import numpy as np
import matplotlib.pyplot as plt
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer


def parse_replay(replay):
  states = []
  x_locations = []
  y_locations = []
  memory = replay.memory.data if isinstance(replay, PrioritizedReplayBuffer) else replay.memory
  for transition in memory:
    transition = transition[-1]  # n-step to single transition
    states.append(transition['next_state'])
    x_locations.append(transition['extra_info']['player_x'])
    y_locations.append(transition['extra_info']['player_y'])
  return states, x_locations, y_locations


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
  # return x, y, pv, ov
