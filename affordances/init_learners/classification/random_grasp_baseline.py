import numpy as np 

class RandomGraspBaseline:
  def score(self, states: np.ndarray) -> np.ndarray:
    n_states = len(states)
    scores = np.zeros(n_states)
    pick = np.random.randint(0, n_states)
    scores[pick] = 1.0
    return scores

  def update(self, initiation_gvf=None):
    pass 

  def add_trajectory(self, transitions, success):
    pass 

    