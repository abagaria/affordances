import numpy as np 

class RandomGraspBaseline:
  def score(self, states: np.ndarray) -> np.ndarray:
    n_states = len(states)
    return np.random.rand(n_states)

  def update(self, initiation_gvf=None):
    pass 

  def add_trajectory(self, transitions, success):
    pass 

    