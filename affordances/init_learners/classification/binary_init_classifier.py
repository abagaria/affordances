import ipdb
import torch
import itertools
import numpy as np

from affordances.utils import utils
from affordances.init_learners.classification.init_classifier import InitiationClassifier
from affordances.init_learners.classification.conv_binary_classifier import ConvClassifier


class ConvInitiationClassifier(InitiationClassifier):
  def __init__(self,
    device, 
    optimistic_threshold: float,
    pessimistic_threshold: float,
    only_reweigh_negative_examples: bool,
    n_input_channels: int = 1,
    maxlen: int = 1000,
    image_dim : int = 84,
    n_epochs: int = 1
  ):
    self.device = device
    self.n_epochs = n_epochs
    self.image_dim = image_dim
    self.optimistic_threshold = optimistic_threshold
    self.pessimistic_threshold = pessimistic_threshold
    self.n_input_channels = n_input_channels
    self.only_reweigh_negative_examples = only_reweigh_negative_examples
    
    self.classifier = ConvClassifier(
      device=device,
      only_reweigh_negative_examples=only_reweigh_negative_examples,
      threshold=None,
      n_input_channels=n_input_channels,
      image_dim=image_dim
    )

    super().__init__(max_n_trajectories=maxlen)
    print(f'Created clf with thresh={optimistic_threshold}, maxlen={maxlen}')

  def _predict(self, states: np.ndarray, threshold) -> np.ndarray:
    # TODO: This converts lz -> np.array. Do this in one place.
    def extract_frame(state):
      if hasattr(state, '_frames'):
        return state._frames[-1]
      assert isinstance(state, np.ndarray), type(state)
      frame = state[np.newaxis, 0, ...] if state.shape == (2, 84, 84) else state
      assert frame.shape == (1, 84, 84), frame.shape
      return frame
    states = [extract_frame(state) for state in states]
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
        self.classifier = ConvClassifier(
          device=self.device,
          threshold=None,
          n_input_channels=self.n_input_channels,
          image_dim=self.image_dim,
          only_reweigh_negative_examples=self.only_reweigh_negative_examples
        )
        self.classifier.fit(X, Y, initiation_gvf, goal, n_epochs=self.n_epochs)

  def save(self, filename: str):
    torch.save(self.classifier.model.state_dict(), filename)

  def load(self, filename: str):
    self.classifier = ConvClassifier(self.device, None, self.n_input_channels)
    self.classifier.model.load_state_dict(
      torch.load(filename)
    )
