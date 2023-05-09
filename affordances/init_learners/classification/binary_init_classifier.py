import torch
import itertools
import collections
import numpy as np

from affordances.utils import utils
from affordances.init_learners.classification.init_classifier import InitiationClassifier
from affordances.init_learners.classification.conv_binary_classifier import ConvClassifier, MlpClassifier


class BinaryInitiationClassifier(InitiationClassifier):
  def __init__(self,
    device, 
    optimistic_threshold: float,
    pessimistic_threshold: float,
    maxlen: int = 10
  ):
    self.device = device
    self.optimistic_threshold = optimistic_threshold
    self.pessimistic_threshold = pessimistic_threshold
  
    super().__init__(max_n_trajectories=maxlen)

  def _predict(self, states: np.ndarray, threshold) -> np.ndarray:
    state_tensor = utils.tensorfy(states, self.device)
    preprocessed_tensor = self.classifier.preprocess_batch(state_tensor)
    predictions = self.classifier.predict(preprocessed_tensor, threshold)
    predictions = predictions.cpu().numpy()
    if predictions.shape == (1, 1):
      return predictions.squeeze()
    return predictions

  def optimistic_predict(self, states: np.ndarray) -> np.ndarray:
    return self._predict(states, self.optimistic_threshold)

  def pessimistic_predict(self, states: np.ndarray) -> np.ndarray:
    return self._predict(states, self.pessimistic_threshold)

  def score(self, states: np.ndarray) -> np.ndarray:
    state_tensor = utils.tensorfy(states, self.device)
    preprocessed_tensor = self.classifier.preprocess_batch(state_tensor)
    scores = self.classifier.predict_probs(preprocessed_tensor)
    scores = scores.cpu().numpy()
    return scores.reshape(-1)

  def add_trajectory(self, trajectory, success_label):
    infos = [transition[-1] for transition in trajectory]
    observations = [transition[-2] for transition in trajectory]

    examples = [(obs, info) for obs, info in zip(observations, infos)]

    if success_label:
      self.positive_examples.append(examples)
    else:
      self.negative_examples.append(examples)

  def update(self, initiation_gvf=None, goal=None):
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
        # TODO (ba): assumes low-dim
        self.classifier = MlpClassifier(self.device, self.input_dim) 
        self.classifier.fit(X, Y, initiation_gvf, goal)

  def save(self, filename: str):
    torch.save(self.classifier.model.state_dict(), filename)

  def load(self, filename: str):
    raise NotImplementedError

class ConvInitiationClassifier(BinaryInitiationClassifier):
  def __init__(device, 
    optimistic_threshold: float,
    pessimistic_threshold: float,
    n_input_channels: int = 1,
    maxlen: int = 10
  ): 
    self.classifier = ConvClassifier(device, None, n_input_channels)
    self.n_input_channels = n_input_channels
    super().__init__(device, 
      optimistic_threshold, 
      pessimistic_threshold, 
      maxlen=maxlen
    )

  def load(self, filename: str):
    self.classifier = ConvClassifier(self.device, None, self.n_input_channels)
    self.classifier.model.load_state_dict(
      torch.load(filename)
    )

class MlpInitiationClassifier(BinaryInitiationClassifier):
  def __init__(self,
    device, 
    optimistic_threshold: float,
    pessimistic_threshold: float,
    input_dim: int = 1,
    maxlen: int = 10
  ):
    self.classifier = MlpClassifier(device, input_dim)
    self.input_dim = input_dim
    super().__init__(device, 
      optimistic_threshold,
      pessimistic_threshold,
      maxlen=maxlen
    )

  def load(self, filename: str):
    self.classifier = MlpClassifier(self.device, self.input_dim)
    self.classifier.model.load_state_dict(
      torch.load(filename)
    )



