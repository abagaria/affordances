import os 
import argparse
import pickle 
import numpy as np 
import pandas as pd

import gzip
import matplotlib.pyplot as plt 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  args = parser.parse_args()

  data = pickle.load(gzip.open(args.path, 'rb+'))
  losses = data.pop('classifier_loss')
  scores = data.pop('scores')
  counts = data.pop('counts')
  success = data.pop('grasp_success')
  rates = np.array(list(success.values())) / np.array(list(counts.values()))

  # df = pd.DataFrame(data)
  # smooth = df.ewm(com=10).mean()

  # plt.plot(smooth['rewards'])
  # plt.title('Return')
  # plt.show()

  # plt.plot(smooth['success'])
  # plt.title('Success')
  # plt.show()

  # plt.plot(losses)
  # plt.title('Loss')
  # plt.show()  
  plt.figure(figsize=(24,10))
  plt.subplot(311)
  plt.bar(counts.keys(), counts.values())
  plt.title('Grasp Attempt Counts')

  plt.subplot(312)
  plt.bar(success.keys(), rates)
  plt.title('Grasp Success Rates')

  plt.subplot(313)
  plt.bar(np.arange(len(scores)), scores)
  plt.title('Final Grasp Scores')

  plt.suptitle(os.path.dirname(args.path).split("/")[-1])
  plt.savefig(os.path.join(os.path.dirname(args.path), 'hist.png'))
  print(os.path.join(os.path.dirname(args.path), 'hist.png'))
  plt.show()


  

	# python -m affordances.utils.plot_curves results/robot_4_10/robot_4_10_081__environmentname_DoorCIP__initlearner_binary__seed_1__optimalik_False/log_seed1.pkl