import torch
import itertools
import collections
import numpy as np

from affordances.utils import utils
from affordances.init_learners.init_learner import InitiationLearner
from affordances.init_learners.classification.conv_binary_classifier import ConvClassifier


class ConvInitiationClassifier(InitiationLearner):
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
    
    self.positive_examples = collections.deque([], maxlen=maxlen)
    self.negative_examples = collections.deque([], maxlen=maxlen)
    self.classifier = ConvClassifier(device, None, n_input_channels)
    
    super().__init__()

  def _predict(self, states: np.ndarray, threshold) -> np.ndarray:
    state_tensor = utils.tensorfy(states, self.device)
    preprocessed_tensor = state_tensor / 255.
    predictions = self.classifier.predict(preprocessed_tensor, threshold)
    predictions = predictions.cpu().numpy()
    if predictions.shape == (1, 1):
      return predictions.squeeze()
    return predictions

  def optimistic_predict(self, states: np.ndarray) -> np.ndarray:
    return self._predict(states, self.optimistic_threshold)

  def pessimistic_predict(self, states: np.ndarray) -> np.ndarray:
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
