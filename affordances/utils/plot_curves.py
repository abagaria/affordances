import os 
import argparse
import pickle 
import pandas as pd

import gzip
import matplotlib.pyplot as plt 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str)
  args = parser.parse_args()

  data = pickle.load(gzip.open(args.path, 'rb+'))
  df = pd.DataFrame(data)
  smooth = df.ewm(com=10).mean()

  plt.plot(smooth['rewards'])
  plt.show()

  plt.plot(smooth['success'])
  plt.show()

	