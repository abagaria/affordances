import ipdb
import numpy as np
import matplotlib.pyplot as plt


def visualize_initiation_table(
    init_table,
    options,
    plot_dir,
    init_type,
    episode
  ):
  x, y, z = [], [], []
  
  for i, option in enumerate(options):
    for position in init_table:
      x.append(position[0])
      y.append(position[1])
      z.append(np.mean(init_table[position][str(option)]))

    plt.subplot(3, 2, i+1)
    plt.scatter(x, y, c=z)
    plt.colorbar()
    plt.title(f'{option} g={option.subgoal_info["player_pos"]}')

  plt.savefig(f'{plot_dir}/{init_type}_init_episode_{episode}.png')
  plt.close()
