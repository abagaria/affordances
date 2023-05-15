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
sns.set(style='whitegrid')
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# sns.set_theme(context='poster', style='whitegrid', palette='colorblind')

OPTIMAL_IK = False
EVAL_FREQ = 10 
ACC_EVAL_FREQ = 250 if not OPTIMAL_IK else 50 
RUN_LENS = {'DoorCIP': 5001, 'SlideCIP':10001, 'LeverCIP': 5001}

CONDITIONS={
                'Baseline Random': 
                    {
                        'init_learner': 'random',
                        'uncertainty':'none'
                    },
                'Baseline Binary':
                    {
                        'init_learner': 'binary',
                        'uncertainty':'none'
                    }, 
                'GVF':
                    {
                        'init_learner': 'gvf',
                        'uncertainty':'none'
                    },
                'Weighted':
                    {
                        'init_learner': 'weighted-binary',
                        'uncertainty':'none'
                    },   
                'Optimistic Binary':
                    {
                        'init_learner': 'binary',
                        'uncertainty':'count_qpos'
                    }, 
                'Optimistic GVF':
                    {
                        'init_learner': 'gvf',
                        'uncertainty':'count_qpos'
                    },
                'Optimistic Weighted':
                    {
                        'init_learner': 'weighted-binary',
                        'uncertainty':'count_qpos'
                    }, 
}
CONDITION_NAMES = list(CONDITIONS.keys())
CONDITION_NAMES.reverse()

y_var = 'Success Rate'
# y_var = 'accuracy'
# y_var = 'size'

# TASKS = ["DoorCIP", "LeverCIP", "DrawerCIP", "SlideCIP"]
# TASKS = ["DoorCIP", "LeverCIP", "SlideCIP"]
# TASKS = ["LeverCIP", "SlideCIP" ]
TASKS = ["DoorCIP", "LeverCIP", "SlideCIP"]
# TASKS=["LeverCIP"]
# TASKS=["SlideCIP"]
# mask = {"optimal_ik":False,
#         "environment_name": "Door",
#         "init_learner":"binary"}
# mask={"sampler":"max", "init_learner":"gvf"}
# mask={"sampler":"sum"}
mask={}

def get_data(rootDirs, conditions=None, task=None, smoothen=100):
  scores = {}
  runs = []
  count=0
  if conditions is not None:
    for condition in conditions.keys():
      scores[condition] = []

  for rootDir in rootDirs:
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

        max_length = RUN_LENS[job_data['environment_name']]
        eval_successes = eval_successes[:max_length]
        eval_rewards = eval_rewards[:max_length]


        # parse reward
        log_df = pd.DataFrame()
        log_df['Success Rate'] = [np.mean(eval_successes)]
        log_df['tag']=str(count)

        if 'uncertainty' not in job_data.keys():
          job_data['uncertainty'] = 'none'

        if 'only_reweigh_negatives' not in job_data.keys():
          job_data['only_reweigh_negatives'] = False

        if 'bonus_scale' not in job_data.keys():
          job_data['bonus_scale'] = 1

        if 'gestation' not in job_data.keys():
          job_data['gestation'] = 0

        if job_data['init_learner'] == 'random':
          job_data['sampler'] = 'sum'

        # fill in log_df
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

        count+=1

  data = pd.concat(runs)
  data = data.astype({"bonus_scale" : str})
  return data, scores

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('path', nargs='+', default=[])
  args = parser.parse_args()

  i=1
  for task in TASKS:
    data, scores = get_data(args.path, conditions=CONDITIONS,task=task)  
    for key, val in mask.items():
      data = data[ data[key] == val ]

    
    i+=1
    plt.figure(figsize=(15,8))
    g = sns.barplot(x='condition',
                y=y_var, 
                data=data,
                errorbar="se",
                order=CONDITION_NAMES
    )
    plt.xlabel("")
    plt.title(task[:-3])
    plt.savefig(f'{args.path[0]}/barplot_{task}.png', bbox_inches='tight')
    plt.show()
  # g.set_titles(col_template = '{col_name}')
    

  # Adjust the spacing between subplots and legend
  # plt.subplots_adjust(bottom=0.22)

  # sns.move_legend(g, "lower center", bbox_to_anchor=[0.5, -0.1],
  # sns.move_legend(g, "lower center", bbox_to_anchor=[0.5, -0.0],
  #               ncol=len(CONDITIONS.keys()), title=None, frameon=False,)


  # Adjust the legend position for each subplot
  # for ax in g.axes.flat:
  #   ax.legend().remove()
  
  # # Create a single legend centered below the subplots
  # legend_handles, legend_labels = g.axes.flat[0].get_legend_handles_labels()
  # legend = g.fig.legend(legend_handles, legend_labels,
  #                     bbox_to_anchor=(0.5, -0.1), 
  #                     loc='lower center', ncol=len(conditions.keys()),
  #                     frameon=False)

  # # Remove the original legend from the figure
  # g.fig.get_axes()[0].legend_.remove()

  

  # plt.savefig(f'{args.path[0]}/{y_var}.png', bbox_inches='tight')
  # plt.show()
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