import ipdb
import matplotlib.pyplot as plt
import numpy as np


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


def get_initiation_stats(
    ground_truth_table,
    measured_init_table,
    filter_length=None
):
  s_o_agreements = []
  mc_results = []
  measured_results = []

  # precisions = []
  # recalls = []
  # f1scores = []

  for start_state in ground_truth_table:
    assert start_state in measured_init_table
    for option in ground_truth_table[start_state]:
      assert option in measured_init_table[start_state]
      ground_truth_measurements = ground_truth_table[start_state][option]
      classifier_predictions = measured_init_table[start_state][option]
      f = lambda x: x.item() if isinstance(x, np.ndarray) else x
      ground_truth_measurements = np.array(list(map(f, ground_truth_measurements)))
      classifier_predictions = np.array(list(map(f, classifier_predictions)))
      assert len(ground_truth_measurements) == len(classifier_predictions)
      agreements = ground_truth_measurements == classifier_predictions
      if filter_length is None or len(agreements) == filter_length:
        s_o_agreements.append(agreements)
        mc_results.append(ground_truth_measurements.tolist())
        measured_results.append(classifier_predictions.tolist())
        # report = classification_report(
        #   ground_truth_measurements.tolist(),
        #   classifier_predictions.tolist(),
        #   output_dict=True, zero_division=1
        # )
        # try:
        #   precisions.append(report['True']['precision'])
        #   recalls.append(report['True']['recall'])
        #   f1scores.append(report['True']['f1-score'])
        # except:
        #   ipdb.set_trace()

  agreement_curves = np.array(s_o_agreements)
  mc_curves = np.array(mc_results)
  
  # precisions = np.array(precisions)[np.newaxis, ...]
  # recalls = np.array(recalls)[np.newaxis, ...]
  # f1scores = np.array(f1scores)[np.newaxis, ...]

  measured_init_curves = np.array(measured_results)

  return agreement_curves, mc_curves #, precisions, recalls, f1scores
  
