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

  df = pd.DataFrame(data)
  smooth = df.ewm(com=10).mean()

  plt.plot(smooth['rewards'])
  plt.title('Return')
  plt.show()

  plt.plot(smooth['success'])
  plt.title('Success')
  plt.show()

  plt.plot(losses)
  plt.title('Loss')
  plt.show()  

  plt.subplot(121)
  plt.bar(np.arange(len(scores)), scores)
  plt.title('Grasp Scores')

  plt.subplot(122)
  plt.bar(counts.keys(), counts.values())
  plt.title('Grasp Counts')
  plt.show()


  

	