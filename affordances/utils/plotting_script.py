import os
import gzip
import glob
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt


def simple_get_log_files(base_dir, jobname, label):
  # Use this when there are no hyperparameter sweeps
  folders = glob.glob(f'{base_dir}/{jobname}*')
  files = [glob.glob(f'{folder}/*/log.pkl')[0] for folder in folders]
  return {label: files}


def get_log_files(base_dir, jobname, hyper2vals, version='new'):
  """Return a dictionary mapping hyperparams to log files."""
  # Example hyper2vals: 
  # {'lr': ['1e-4', '3e-4', '6.25e-5'], 'sigma': ['0.1', '0.5', '1.0']}
  # Example output:
  # {'__lr_1e-4__sigma_0.1': [path2seed0pkl, path2seed1pkl, ...], ...}

  def get_keys(hyper2vals):
    # hyper2vals: maps hyper name to list of hyper values

    names = list(hyper2vals.keys())
    patterns = []

    for values in itertools.product(*hyper2vals.values()):
      patterns.append(
        ''.join([f'__{name}_{val}' for name, val in zip(names, values)])
      )

    return patterns

  def values(keys, folders):
    key2val = {}
    for pattern in keys:
      key2val[pattern] = [f for f in folders if pattern in f]
    return key2val
  
  def pattern2files(pattern2folders):
    p2fs = {}
    for pattern in pattern2folders:
      folders = pattern2folders[pattern]
      fname = 'log_seed*.pkl' if version == 'new' else 'log.pkl'
      glob_pattern = lambda folder: f'{folder}/{fname}' if version == 'new' else f'{folder}/*/{fname}'
      # import ipdb; ipdb.set_trace()
      files = [glob.glob(glob_pattern(folder)) for folder in folders][0]
      p2fs[pattern] = files
    return p2fs
  
  folders = glob.glob(os.path.join(base_dir, jobname) + '*')
  print(folders)
  hyper_patterns = get_keys(hyper2vals)
  print('hyper_patterns: ', hyper_patterns)
  pattern2folders = values(hyper_patterns, folders)
  print(pattern2folders)
  return pattern2files(pattern2folders)


def get_scores(pattern2files):
  """Return a dictionary mapping hyperparam patters to reward lists."""
  pattern2scores = {}
  for pattern, filenames in pattern2files.items():
    scores = [pickle.load(gzip.open(name, 'rb'))['rewards'] for name in filenames]
    pattern2scores[pattern] = scores
  return pattern2scores


def truncate(scores, max_length=-1, min_length=-1):
  filtered_scores = [score_list for score_list in scores if len(score_list) > min_length]
  if not filtered_scores:
    return filtered_scores
  lens = [len(x) for x in filtered_scores]
  print('lens: ', lens)
  min_length = min(lens)
  if max_length > 0:
    min_length = min(min_length, max_length)
  truncated_scores = [score[:min_length] for score in filtered_scores]
  
  return truncated_scores


def get_plot_params(array):
  median = np.median(array, axis=0)
  means = np.mean(array, axis=0)
  std = np.std(array, axis=0)
  N = array.shape[0]
  top = means + (std / np.sqrt(N))
  bot = means - (std / np.sqrt(N))
  return median, means, top, bot


def moving_average(a, n=25):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n-1:] / n


def smoothen_data(scores, n=10):
  smoothened_cols = scores.shape[1] - n + 1
  smoothened_data = np.zeros((scores.shape[0], smoothened_cols))
  for i in range(scores.shape[0]):
      smoothened_data[i, :] = moving_average(scores[i, :], n=n)
  return smoothened_data


def generate_plot(score_array, label, smoothen=0, linewidth=2, all_seeds=False):
  # smoothen is a number of iterations to average over
  if smoothen > 0:
      score_array = smoothen_data(score_array, n=smoothen)
  median, mean, top, bottom = get_plot_params(score_array)
  plt.plot(mean, linewidth=linewidth, label=label)
  plt.fill_between( range(len(top)), top, bottom, alpha=0.2 )
  if all_seeds:
    print(score_array.shape)
    for i, score in enumerate(score_array):
      plt.plot(score, linewidth=linewidth, label=label+f"_{i+1}")

    
if __name__ == '__main__':
  rainbow_expname = 'tune_rainbow_minigrid_4rooms_including_bonus'
  rainbow_jname = 'tune_rainbow'

  # dsc_expname = 'dsc_epsilon_schedule_sweep'
  dsc_expname = 'dsc_with_bonus_epsilon_schedule_sweep'
  # dsc_expname = 'her_with_bonus_epsilon_schedule_sweep'
  # dsc_jname = 'dsc_exploration'
  dsc_jname = 'dscexploration'
  # dsc_jname = 'herbonus2'

  base_dir = os.path.expanduser(f'~/Downloads/affordances_logs/{dsc_expname}')
  jname = dsc_jname

  rainbow_hyper_dict = {
    'lr': ['1e-4', '3e-4', '6.25e-5'],
    'sigma': ['0.1', '0.5', '1.0'],
    'bonusscale': ['1e-3', '0']
  }

  dsc_hyper_dict = {
    # 'epsilondecaysteps': ['12500', '25000', '37500', '50000']
    'epsilondecaysteps': ['1000', '5000', '10000'],
    'explorationbonusscale': ['1e-2', '1e-3']
  }

  pattern_to_files = get_log_files(base_dir, jname, dsc_hyper_dict)
  # pattern_to_files = simple_get_log_files(base_dir, jname, 'DSC')
  print('pattern2files:', pattern_to_files)
  
  pattern_to_score_lists = get_scores(pattern_to_files)

  for pattern, score_lists in pattern_to_score_lists.items():
    score_lists = truncate(score_lists)
    score_array = np.asarray(score_lists)
    label = pattern[2:] if pattern[:2] == '__' else pattern
    generate_plot(score_array, label, smoothen=10, all_seeds=False)

  plt.legend()
  plt.show()
