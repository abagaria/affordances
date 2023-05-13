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

  # experiments = ['init_classifier_g10_new_starts_sigmoid_reweigh_both',
  #                'init_classifier_g10_new_starts_sigmoid',
  #                'vanilla_binary_init_classifier_g10_new_starts_sigmoid',
  #                'init_classifier_g10_new_starts_always_update',]
                #  'gvf_init_classifier_g20_new_starts_sigmoid2',
                #  'gvf_init_classifier_g20_new_starts_sigmoid_competence_dev']

  # gestation_period_10
  experiments = [
    'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_10__nclassifiertrainingepochs_1',  # good  # Good
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_10__nclassifiertrainingepochs_2',  # good
  'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_100__nclassifiertrainingepochs_1',  # good  # Good
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_10__nclassifiertrainingepochs_3',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_100__nclassifiertrainingepochs_2',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_100__nclassifiertrainingepochs_3',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_1000__nclassifiertrainingepochs_1',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_1000__nclassifiertrainingepochs_2',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_competence__nclassifiertrainingtrajectories_1000__nclassifiertrainingepochs_3',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_10__nclassifiertrainingepochs_1',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_10__nclassifiertrainingepochs_2',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_10__nclassifiertrainingepochs_3',  # good
  'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_100__nclassifiertrainingepochs_1',  # good  # Good
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_100__nclassifiertrainingepochs_2',  # good
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_100__nclassifiertrainingepochs_3',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_1000__nclassifiertrainingepochs_1',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_1000__nclassifiertrainingepochs_2',
# 'monte_clf_sweep2__gestationperiod_10__uncertaintytype_none__nclassifiertrainingtrajectories_1000__nclassifiertrainingepochs_3',
  'monte_vanilla_binary_init_classifier',

  ## GVF
  'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_10__uncertaintytype_competence__optimisticthreshold_0.4',    # good
  #   # 'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_10__uncertaintytype_competence__optimisticthreshold_0.5',
  #   # 'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_10__uncertaintytype_competence__optimisticthreshold_0.6',
  #   # 'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_10__uncertaintytype_competence__optimisticthreshold_0.7',
    'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_10__uncertaintytype_none__optimisticthreshold_0.4',        # good
  #   # 'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_10__uncertaintytype_none__optimisticthreshold_0.5',
  #   # 'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_10__uncertaintytype_none__optimisticthreshold_0.6',
  #   # 'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_10__uncertaintytype_none__optimisticthreshold_0.7',
  #   'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_20__uncertaintytype_competence__optimisticthreshold_0.4',
  #   'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_20__uncertaintytype_competence__optimisticthreshold_0.5',
  #   'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_20__uncertaintytype_competence__optimisticthreshold_0.6',
  #   'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_20__uncertaintytype_competence__optimisticthreshold_0.7',
  #   'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_20__uncertaintytype_none__optimisticthreshold_0.4',
  #   'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_20__uncertaintytype_none__optimisticthreshold_0.5',
  #   'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_20__uncertaintytype_none__optimisticthreshold_0.6',
  #   'monte_gvf_threshold_gestation_uncertainty_sweep__gestationperiod_20__uncertaintytype_none__optimisticthreshold_0.7',
  ]
  

  # Minigrid results
  # experiments = [
  #   'minigrid_4rooms_gvf_threshold0_4_uncertainty_none_g10',
  #   'minigrid_4rooms_gvf_threshold0_4_uncertainty_competence_g10',
  #   # 'minigrid_4rooms_init_clf_sweep3__uncertaintytype_competence__-only_reweigh_negative_examples__-use_weighted_classifiers',
  #   'minigrid_4rooms_init_clf_sweep3__uncertaintytype_none__+only_reweigh_negative_examples__+use_weighted_classifiers',  # Good
  #   # 'minigrid_4rooms_init_clf_sweep3__uncertaintytype_competence__+only_reweigh_negative_examples__-use_weighted_classifiers',
  #   # 'minigrid_4rooms_init_clf_sweep3__uncertaintytype_none__-only_reweigh_negative_examples__+use_weighted_classifiers',
  #   'minigrid_4rooms_init_clf_sweep3__uncertaintytype_none__-only_reweigh_negative_examples__-use_weighted_classifiers',
  #   # 'minigrid_4rooms_init_clf_sweep3__uncertaintytype_competence__-only_reweigh_negative_examples__+use_weighted_classifiers',
  #   'minigrid_4rooms_init_clf_sweep3__uncertaintytype_competence__+only_reweigh_negative_examples__+use_weighted_classifiers',  # Good
  #   # 'minigrid_4rooms_init_clf_sweep3__uncertaintytype_none__+only_reweigh_negative_examples__-use_weighted_classifiers',
  # ]
  
  plt.figure(figsize=(20, 12))
  
  for experiment in experiments:
    print(experiment)
    data_dir = f'results/{experiment}'
    
    gt_table, measured_init_table = load_tables(args.log_dir, args.episode, experiment, args.seed)
    accuracy, mc = get_initiation_stats(gt_table, measured_init_table, filter_length=args.episode+1)
    generate_plot(accuracy, mc, experiment)

  plt.legend()
  plt.savefig(f'{args.results_dir}/mc_measured_agreement_montezumasrevenge.png')
  plt.close()
