"""Misc utils."""

import os
import gzip
import torch
import pickle
import itertools
import numpy as np


def rand_argmax(x):
  """A random tie-breaking argmax."""
  return np.argmax(
    np.random.random(x.shape) * (x == x.max())
  )


def tensorfy(x, device, transpose_to=None):
  if isinstance(x, list):
    x = np.asarray(x)
  if transpose_to is not None:
    x = x.transpose(transpose_to)
  if isinstance(x, np.ndarray):
    x = torch.as_tensor(x).contiguous().float()
  if len(x.shape) == 2:  # Add batch dim
    x = x.unsqueeze(0)
  return x.to(device)


def unpack_transitions(transitions):
  states = []
  actions = []
  rewards = []
  next_states = []
  dones = []
  for transition in transitions:
    states.append(transition['state'].transpose((1, 2, 0)))
    actions.append(transition['action'])
    rewards.append(transition['reward'])
    next_states.append(transition['next_state'].transpose((1, 2, 0)))
    dones.append(transition['is_state_terminal'])
  return states, actions, rewards, next_states, dones


def flatten(x, to_list=True):
  x = itertools.chain.from_iterable(x)
  return list(x) if to_list else x


def create_log_dir(experiment_name):
  path = os.path.join(os.getcwd(), experiment_name)
  try:
      os.makedirs(path, exist_ok=True)
  except OSError:
      pass
  else:
      print("Successfully created the directory %s " % path)
  return path


def safe_zip_write(filename, data):
  """Safely writes a file to disk.
  Args:
    filename: str, the name of the file to write.
    data: the data to write to the file.
  """
  filename_temp = f"{filename}.tmp.gz"
  with gzip.open(filename_temp, 'wb+') as f:
    pickle.dump(data, f)
  os.replace(filename_temp, filename)


def parse_log_file(experiment_name, sub_dir, seed):
  fname = os.path.join('logs', experiment_name, sub_dir, str(seed), 'log.pkl')
  with gzip.open(fname, 'rb') as f:
    log_dict = pickle.load(f)
  return log_dict


def set_random_seed(seed):
  import torch
  import random
  import numpy as np
  
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def softmax(scores, temperature):
  from scipy import special
  assert temperature > 0, temperature
  return special.softmax(scores / temperature)
