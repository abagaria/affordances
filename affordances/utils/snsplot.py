import os 
import gzip
import pickle 
import argparse

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from affordances.utils.plotting_script import generate_plot, truncate, moving_average

sns.set_palette('colorblind')
sns.set(font_scale=1.5)

EVAL_FREQ = 10 
TASKS = ["DoorCIP", "LeverCIP", "DrawerCIP", "SlideCIP"]
conditions = { 
                'Random': 
                    {
                        'init_learner': 'random',
                    },

                'Binary':
                    {
                        'init_learner': 'binary',
                    },   
            }
# mask = {"environment_name":"Door"}
# mask = {}

def get_data(rootDir, conditions=None, task=None, smoothen=10):
  scores = {}
  runs = []
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

      print('Found run: %s' % dirName)

      # if we find a log, read into dataframe
      path = os.path.join(dirName,fname)
      eval_files = pickle.load(gzip.open(path, "rb+"))
      eval_successes = eval_files['success']
      eval_rewards = eval_files['rewards']

      log_df = pd.DataFrame()
      log_df['success'] = moving_average(eval_successes, n=smoothen)
      log_df['reward'] = moving_average(eval_rewards, n=smoothen)
      log_df['episode'] = np.arange(len(log_df)) + EVAL_FREQ
      log_df['tag']=path
      for key, val in job_data.items():
        if key == "environment_name":
          job_data[key] = job_data[key][:-3]
        log_df[key] = job_data[key]

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

  data = pd.concat(runs)
  return data, scores

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  args = parser.parse_args()

  data, scores = get_data(args.path, conditions=conditions)
  y_var = "success"
  g = sns.relplot(x='episode',
                  y=y_var, 
                  kind='line',
                  data=data,
                  alpha=0.8,
                  hue="condition",
                  hue_order=conditions.keys(),
                  col="environment_name",
                  facet_kws={"sharex":False,"sharey":True},
                  errorbar="se"
  )
  g.set_titles(col_template = '{col_name}')
  g.set_axis_labels( "Episode" , y_var)
  plt.savefig(f'{args.path}/{y_var}.svg')
  plt.savefig(f'{args.path}/{y_var}.png')
  plt.show()
  

  # Akhil's logic: 
  # for i, task in enumerate(TASKS):
  #   scores =get_data(args.path, conditions=conditions, task=task)
  #   for pattern, score_lists in scores.items():
  #     score_lists = truncate(score_lists, max_length=500)
  #     score_array = np.asarray(score_lists)
  #     label = pattern[2:] if pattern[:2] == '__' else pattern
  #     generate_plot(score_array, label, smoothen=10, all_seeds=False)

  #   plt.title(task)
  #   plt.legend()
  #   plt.savefig(os.path.join(args.path, f'{task}.png'))
  #   plt.show()   