import ipdb
import collections
import numpy as np
import matplotlib.pyplot as plt


def parse_replay(replay):
  states = []
  x_locations = []
  y_locations = []
  for transition in replay.memory:
    transition = transition[-1]  # n-step to single transition
    states.append(transition['next_state'])
    x_locations.append(transition['extra_info']['player_x'])
    y_locations.append(transition['extra_info']['player_y'])
  return np.array(states), np.array(x_locations), np.array(y_locations)


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
    value_list = np.array(value_chunk).tolist()
    values.extend(value_list)
    for x, y, value in zip(x_chunk, y_chunk, value_list):
      value_dict[(x, y)].append(value)

  x = [pos[0] for pos in value_dict]
  y = [pos[1] for pos in value_dict]
  v = [np.mean(value_dict[pos]) for pos in value_dict]
  e = [np.std(value_dict[pos]) for pos in value_dict]

  plt.figure(figsize=(20,12))
  plt.subplot(121)
  plt.scatter(x, y, c=v, s=50)
  plt.colorbar()
  plt.title('Value Mean')
  plt.subplot(122)
  plt.scatter(x, y, c=e, s=50)
  plt.colorbar()
  plt.title('Value Std Dev')
  plt.savefig(f'plots/{experiment_name}/{seed}/vf_{episode}.png')
  plt.close()
