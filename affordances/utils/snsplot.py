import os 
import gzip
import pickle 
import argparse

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from affordances.utils.plotting_script import generate_plot, truncate, moving_average
from affordances.utils.conditions_dict import conditions

sns.set_palette('colorblind')
sns.set(font_scale=1.5)

# IK = False
EVAL_FREQ = 10 
# TASKS = ["DoorCIP", "LeverCIP", "DrawerCIP", "SlideCIP"]
TASKS = ["DoorCIP", "LeverCIP", "SlideCIP"]
# mask = {"optimal_ik":False,
#         "environment_name": "Door",
#         "init_learner":"binary"}
mask = {}

def get_data(rootDir, conditions=None, task=None, smoothen=100):
  scores = {}
  runs = []
  count=0
  if conditions is not None:
    for condition in conditions.keys():
      scores[condition] = []

  for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
      if "log_seed" not in fname:
        continue 
      
      # process the job config
      config_path = os.path.join(dirName, 'config.pkl')
      with gzip.open(config_path, 'rb+') as f:
        job_data = pickle.load(f)

      if task is not None:
        if job_data['environment_name'] != task:
          continue

      # print('Found run: %s' % dirName)

      # if we find a log, read into dataframe
      path = os.path.join(dirName,fname)
      eval_files = pickle.load(gzip.open(path, "rb+"))
      eval_successes = eval_files['success']
      eval_rewards = eval_files['rewards']
      # print(len(eval_successes))

      log_df = pd.DataFrame()
      log_df['success'] = moving_average(eval_successes, n=smoothen)
      log_df['reward'] = moving_average(eval_rewards, n=smoothen)
      log_df['episode'] = np.arange(len(log_df)) + EVAL_FREQ
      log_df['tag']=str(count)
      for key, val in job_data.items():
        if key == "environment_name":
          job_data[key] = job_data[key][:-3]
        log_df[key] = job_data[key]

      if 'uncertainty' not in job_data.keys():
        job_data['uncertainty'] = 'none'
        log_df['uncertainty'] = 'none'

      # maybe filter on conditions 
      if conditions is not None: 
        for cond_key, cond_dict in conditions.items(): 
          if np.all( [job_data[k] == v for k, v in cond_dict.items()] ):
            scores[cond_key].append(eval_successes)
            log_df['condition'] = cond_key
            runs.append(log_df)
            break
      else:
        runs.append(log_df)

      count+=1

  data = pd.concat(runs)
  return data, scores

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  args = parser.parse_args()

  for task in TASKS:

    data, scores = get_data(args.path, conditions=conditions, task=task)
    for key, val in mask.items():
      data = data[ data[key] == val ]

    y_var = "success"
    g = sns.relplot(x='episode',
                    y=y_var, 
                    kind='line',
                    data=data,
                    alpha=0.8,
                    hue="condition",
                    hue_order=conditions.keys(),
                    # col="environment_name",
                    # row="segment",
                    # style="optimal_ik",
                    # style='sampler',
                    col="sampler",
                    row="uncertainty",
                    facet_kws={"sharex":False,"sharey":True},
                    errorbar="se"
    )
    plt.suptitle(task)
    # g.set_titles(col_template = '{col_name}')
    g.set_axis_labels( "Episode" , y_var)
    # plt.savefig(f'{args.path}/{y_var}.svg')
    plt.savefig(f'{args.path}/{y_var}.png', bbox_inches='tight')
    plt.show()


  # Akhil's logic: 
  # ideal_len = 4991
  # for i, task in enumerate(TASKS):

  #   data, scores =get_data(args.path, conditions=conditions, task=task)
  #   for pattern, score_lists in scores.items():

  #     try:
  #       min_length = min([len(arr) for arr in score_lists])
  #       score_lists = truncate(score_lists, max_length=min_length)
  #       # score_lists = [s for s in score_lists if len(s) == ideal_len]
  #       score_array = np.asarray(score_lists)
  #       label = pattern[2:] if pattern[:2] == '__' else pattern
  #       generate_plot(score_array, label, smoothen=100, all_seeds=False)
  #     except Exception as e:
  #       print(e)

  #   plt.title(task)
  #   plt.legend()
  #   plt.savefig(os.path.join(args.path, f'{task}.png'))
  #   plt.show()   