import ipdb
import gzip
import pickle
import argparse
import numpy as np

from matplotlib import pyplot as plt
import affordances.utils.plotting_script as plotting_utils
from affordances.utils.init_accuracy_plotting import get_initiation_stats


def load_tables(log_dir, episode, experiment_name, seed):
  gt_fname = f'{log_dir}/{experiment_name}/episode_{episode}_gt_seed{seed}.pkl'
  init_fname = f'{log_dir}/{experiment_name}/episode_{episode}_measured_seed{seed}.pkl'
  with gzip.open(gt_fname, 'rb') as f:
    ground_truth_table = pickle.load(f)
  with gzip.open(init_fname, 'rb') as f:
    measured_init_table = pickle.load(f)
  return ground_truth_table, measured_init_table


def generate_plot(agreement_curves, mc_curves, experiment_name):
  
  plt.subplot(1, 3, 1)
  plt.plot(agreement_curves.mean(axis=0), label=experiment_name)
  # print(agreement_curves.shape)
  # plt.bar(x=range(agreement_curves.shape[1]), height=agreement_curves.mean(), label=experiment_name)
  plt.title('Accuracy')
  # plt.xlabel('Episode')
  
  plt.subplot(1, 3, 2)
  plt.plot(mc_curves.mean(axis=0), label=experiment_name)
  plt.title('Monte-Carlo Estimate')
  plt.xlabel('Episode')
  
  # plt.subplot(1, 3, 3)
  # plotting_utils.generate_plot(precisions, 'precision', smoothen=10)
  # plotting_utils.generate_plot(recalls, 'recall', smoothen=10)
  # plotting_utils.generate_plot(f1scores, 'F1-Score', smoothen=10)
  # plt.legend()
  # plt.xlabel('Episode')
  # plt.title('Classification Report')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir', type=str, default='/gpfs/data/gdk/abagaria/affordances_logs')
  parser.add_argument("--results_dir", type=str, default='results',
                        help='the name of the directory used to store results')
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--episode', type=int)
  args = parser.parse_args()

  # Minigrid results
  experiments = [
    'minigrid_big_doorkey_vanilla_classifier',
    'minigrid_big_doorkey_dev3',  # GVF, uncertainty=none
    'minigrid_big_doorkey_weighted_classifier_uncertainty_none',
    'minigrid_big_doorkey_weighted_classifier_uncertainty_competence',
    'minigrid_big_doorkey_gvf_competence',
  ]
  
  plt.figure(figsize=(20, 12))
  
  for experiment in experiments:
    print(experiment)
    data_dir = f'results/{experiment}'
    
    gt_table, measured_init_table = load_tables(args.log_dir, args.episode, experiment, args.seed)
    accuracy, mc = get_initiation_stats(gt_table, measured_init_table, filter_length=args.episode+1)
    generate_plot(accuracy, mc, experiment)

  plt.legend()
  plt.savefig(f'{args.results_dir}/mc_measured_agreement_doorKey16x16.png')
  plt.close()
