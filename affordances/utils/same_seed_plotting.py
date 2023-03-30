"""If we do several runs for the same seed, we have to rely on onager tags
to disambiguate between the runs. But that doesn't play well with our plotting
set up, so we do some file manipulation to make it seem like we ran the
experiment with different seeds.
"""

import os
import glob

from affordances.utils import utils


log_dir = os.path.expanduser('/gpfs/data/gdk/abagaria/affordances_logs')


def relabel_seeds(experiment_name: str, jobname: str, tag_args: str):
  """Go into the directories with the different onager tags and rename the log files inside."""
  directories = glob.glob(f'{log_dir}/{experiment_name}/{jobname}_*__{tag_args}')
  for i, directory in enumerate(directories):
    try:
      os.replace(f'{directory}/log_seed42.pkl', f'{directory}/log_seed{i}.pkl')
    except:
      pass


def rm_onager_tag_from_experiment_dir(experiment_name, jobname, tag_args):
  directories = glob.glob(f'{log_dir}/{experiment_name}/{jobname}_*__{tag_args}')
  new_directory = f'{log_dir}/{experiment_name}/{jobname}__{tag_args}'
  utils.create_log_dir(new_directory)
  for i, directory in enumerate(directories):
    os.replace(f'{directory}/log_seed{i}.pkl', f'{new_directory}/log_seed{i}.pkl')

def move_files_to_same_dir(experiment_dir):
  pass


if __name__ == '__main__':
  # jname = 'weighted_classifier_rr_seed42'
  jname = 'her_rr_seed42'
  # expname = 'dsc_weighted_classifier_seed42_random_resets'
  expname = 'her_seed42_random_resets'
  # tag_args = 'explorationbonusscale_0__initclassifiertype_binary'
  tag_args = 'explorationbonusscale_0'
  relabel_seeds(expname, jname, tag_args)
  rm_onager_tag_from_experiment_dir(expname, jname, tag_args)
