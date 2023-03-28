import os 
import gzip
import pickle 
import argparse

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

sns.set_palette('colorblind')
sns.set(font_scale=1.5)

EVAL_FREQ = 10 
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
mask = {}

def get_data(rootDir, conditions=None):
  scores = {}
  runs=[]
  for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
      if "log_seed" not in fname:
        continue 
      
      print('Found run: %s' % dirName)

      # if we find a log, read into dataframe
      path = os.path.join(dirName,fname)
      eval_files = pickle.load(gzip.open(path, "rb+"))
      eval_successes = eval_files['success']
      eval_rewards = eval_files['rewards']

      # process the job config
      config_path = os.path.join(dirName, 'config.pkl')
      with gzip.open(config_path, 'rb+') as f:
        job_data = pickle.load(f)

      log_df = pd.DataFrame()
      log_df['success'] = eval_successes
      log_df['reward'] = eval_rewards
      log_df['episode'] = np.arange(len(log_df)) + EVAL_FREQ
      log_df['tag']=path
      for key, val in job_data.items():
        if key == "environment_name":
          job_data[key] = job_data[key][:-3]
        log_df[key] = job_data[key]

      # compute best-yet success
      # max_success = []
      # max_violation = []
      # best_succ = -1
      # for i, succ in enumerate(log_df['success']):
      #     if succ > best_succ:
      #         best_succ=suc
      #     max_success.append(best_succ)
      # log_df['Success Rate'] = max_success

      # maybe filter on conditions 
      if conditions is not None: 
        for cond_key, cond_dict in conditions.items(): 
          if np.all( [log_df[k] == v for k, v in cond_dict.items()] ):
            log_df['condition'] = cond_key
            runs.append(log_df)
            print(f'found condition {cond_key}')
            break

      else:
        runs.append(log_df)

  data = pd.concat(runs)
  return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    
    data=get_data(args.path, conditions=conditions)
    
    # maybe mask
    for key, val in mask.items():
        data = data[ data[key] == val ]

    # plot
    y_var = "reward"
    # y_var = "reward"
    g = sns.relplot(x='episode',
                    y=y_var, 
                    kind='line',
                    data=data,
                    alpha=0.4,
                    hue="condition",
                    # hue_order=conditions.keys(),
                    col="environment_name",
                    facet_kws={"sharex":False,"sharey":True},
                    )
    g.set_titles(col_template = '{col_name}')
    g.set_axis_labels( "Episode" , y_var)
    # plt.savefig(f'{args.path}/{y_var}.svg')
    plt.savefig(f'{args.path}/{y_var}.png')
    plt.show()