"""Misc utils."""

import torch
import numpy as np


def rand_argmax(x):
  """A random tie-breaking argmax."""
  return np.argmax(
    np.random.random(x.shape) * (x == x.max())
  )


def tensorfy(x, device, transpose_to=None):
  if isinstance(x, list):
    x = np.array(x)
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