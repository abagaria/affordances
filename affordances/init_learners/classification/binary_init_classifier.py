import torch
import itertools
import collections
import numpy as np

from affordances.utils import utils
from affordances.init_learners.classification.init_classifier import InitiationClassifier
from affordances.init_learners.classification.conv_binary_classifier import ConvClassifier


class ConvInitiationClassifier(InitiationClassifier):
  def __init__(self,
    device, 
    optimistic_threshold: float,
    pessimistic_threshold: float,
    n_input_channels: int = 1,
    maxlen: int = 10
  ):
    self.device = device
    self.optimistic_threshold = optimistic_threshold
    self.pessimistic_threshold = pessimistic_threshold
    self.n_input_channels = n_input_channels
    self.classifier = ConvClassifier(device, None, n_input_channels)
    self.option = None # reference to option that owns this classifier. 

    super().__init__(max_n_trajectories=maxlen)

  def _predict(self, states: np.ndarray, threshold) -> np.ndarray:
    state_tensor = utils.tensorfy(states, self.device)
    preprocessed_tensor = state_tensor / 255.
    predictions = self.classifier.predict(preprocessed_tensor, threshold)
    predictions = predictions.cpu().numpy()
    if predictions.shape == (1, 1):
      return predictions.squeeze()
    return predictions

  def optimistic_predict(self, states: np.ndarray, infos: np.ndarray, use_ucb:bool) -> np.ndarray:
    state_tensor = utils.tensorfy(states, self.device)
    preprocessed_tensor = state_tensor / 255.
    predictions = self.classifier.predict_proba(preprocessed_tensor)
    
    assert (self.option != None) 
    assert (len(states) == len(infos))
    assert (len(list(self.option.visitation_counts.keys())) > 0)

    use_ucb = int(use_ucb)

    ucb_predictions = []
    for _, (pred, i) in enumerate(zip(predictions, infos)): 
      N = max(self.option.visitation_counts.get(i['player_pos'], 0), 1)
      ucb_predictions.append(pred.cpu().numpy() + use_ucb/np.sqrt(N))
    ucb_predictions = np.array(ucb_predictions) 
    return ucb_predictions > self.optimistic_threshold

  def pessimistic_predict(self, states: np.ndarray, infos:np.ndarray) -> np.ndarray:
    return self._predict(states, self.pessimistic_threshold)

  def add_trajectory(self, trajectory, success_label):
    infos = [transition[-1] for transition in trajectory]
    observations = [transition[-2] for transition in trajectory]

    examples = [(obs, info) for obs, info in zip(observations, infos)]

    if success_label:
      self.positive_examples.append(examples)
    else:
      self.negative_examples.append(examples)

  def update(self):
    if self.positive_examples and self.negative_examples:
      pos_examples = list(itertools.chain.from_iterable(self.positive_examples))
      neg_examples = list(itertools.chain.from_iterable(self.negative_examples))
      
      pos_observations = [eg[0] for eg in pos_examples]
      neg_observations = [eg[0] for eg in neg_examples]

      positive_labels = torch.ones((len(pos_observations),))
      negative_labels = torch.zeros((len(neg_observations),))

      X = pos_observations + neg_observations
      Y = torch.cat((positive_labels, negative_labels), dim=0)
      if self.classifier.should_train(Y):
        self.classifier.fit(X, Y)

  def save(self, filename: str):
    torch.save(self.classifier.model.state_dict(), filename)

  def load(self, filename: str):
    self.classifier = ConvClassifier(self.device, None, self.n_input_channels)
    self.classifier.model.load_state_dict(
      torch.load(filename)
    )
