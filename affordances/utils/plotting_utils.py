import os
import gzip
import torch
import pickle
import itertools
import numpy as np

fname = "/home/sreehari/Desktop/affordances/logs/4rooms/log_seed2.pkl"
with gzip.open(fname, 'rb') as f:
    log_dict = pickle.load(f)

breakpoint()