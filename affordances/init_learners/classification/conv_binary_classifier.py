from typing import Optional

import torch
import random
import numpy as np
import torch.nn.functional as F

from affordances.utils import utils
from pfrl.nn.atari_cnn import SmallAtariCNN
from torch.utils.data import Dataset, DataLoader
from affordances.init_learners.gvf.init_gvf import GoalConditionedInitiationGVF


class ImageCNN(torch.nn.Module):
  def __init__(self, n_input_channels: int = 3):
    super().__init__()

    self.model = torch.nn.Sequential(
      SmallAtariCNN(n_input_channels, n_output_channels=256),
      torch.nn.Linear(in_features=256, out_features=128),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=128, out_features=1)
    )

  def forward(self, x):
    assert x.max() <= 1.1, (x.max(), x.dtype)
    return self.model(x.float())

class MLP(torch.nn.Module):
  def __init__(self, input_dim):
    super().__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Linear(in_features=input_dim, out_features=256),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=256, out_features=128),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=128, out_features=1) 
    )

  def forward(self, x):
    return self.model(x.float())


class Classifier:
  def __init__(self,
        device,
        threshold=0.5,
        batch_size=32,
        lr=1e-3,
        only_reweigh_negative_examples=True):
    
    self.device = device
    self.is_trained = False
    self.threshold = threshold
    self.batch_size = batch_size
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    self.only_reweigh_negative_examples = only_reweigh_negative_examples

    # Debug variables
    self.losses = []

  @torch.no_grad()
  def predict(self, X, threshold=None):
    logits = self.model(X)
    probabilities = torch.sigmoid(logits)
    threshold = self.threshold if threshold is None else threshold
    return probabilities > threshold

  @torch.no_grad()
  def predict_probs(self, X):
    logits = self.model(X)
    probabilities = torch.sigmoid(logits)
    return probabilities

  def determine_pos_weight(self, y):
    n_negatives = len(y[y != 1])
    n_positives = len(y[y == 1])
    if n_positives > 0:
      pos_weight = (1. * n_negatives) / n_positives
      return torch.as_tensor(pos_weight).float()
  
  @torch.no_grad()
  def determine_instance_weights(
    self,
    states: np.ndarray,
    labels: torch.Tensor,
    init_gvf: GoalConditionedInitiationGVF,
    goal: Optional[np.ndarray]
  ):

    # TODO: (ba) assumes np.float32 states 
    assert isinstance(states, np.ndarray), 'Conversion done in TD(0)'
    assert states.dtype == np.float32, 'Preprocessing done in TD(0)'
    if goal is not None:
      goals = np.repeat(goal[np.newaxis, ...], repeats=len(states), axis=0)
      assert isinstance(goal, np.ndarray), 'Conversion done in TD(0)'
      # assert goals.dtype == np.uint8, 'Preprocessing done in TD(0)'
      values = init_gvf.get_values(states, goals)
    else:
      values = init_gvf.get_values(states)
    
    values = utils.tensorfy(values, self.device)  # TODO(ab): keep these on GPU
    weights = values.clip(0., 1.)
    weights[labels == 0] = 1. - weights[labels == 0]

    if self.only_reweigh_negative_examples:
      values[labels == 1] = 1.
      # else values[labels==1] are already = V(s), which is what we want
    return weights

  def should_train(self, y):
    enough_data = len(y) > self.batch_size
    has_positives = len(y[y == 1]) > 0
    has_negatives = len(y[y != 1]) > 0
    return enough_data and has_positives and has_negatives

  def preprocess_batch(self, X):
    raise NotImplementedError

  def fit(self, X, y, initiation_gvf=None, goal=None, n_epochs=1):
    dataset = ClassifierDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    if self.should_train(y):
      losses = []

      for _ in range(n_epochs):
        epoch_loss = self._train(dataloader, initiation_gvf, goal)                
        losses.append(epoch_loss)
      
      self.is_trained = True

      mean_loss = np.mean(losses)
      self.losses.append(mean_loss)

  def _train(self, loader, initiation_gvf=None, goal=None):
    """ Single epoch of training. """
    weights = None
    batch_losses = []

    for sample in loader:
      sampled_inputs, sampled_labels = sample[0], sample[1]
      sampled_inputs = utils.tensorfy(sampled_inputs, self.device)
      sampled_inputs = self.preprocess_batch(sampled_inputs)
      sampled_labels = sampled_labels.to(self.device)
      
      pos_weight = self.determine_pos_weight(sampled_labels)

      # TODO(ab): This is inefficient: we sample many batches before a valid one
      if not pos_weight:
        continue

      if initiation_gvf is not None:
        weights = self.determine_instance_weights(
          sample[0].numpy(), # DataLoader converts to tensor, undoing that here
          sampled_labels,  # This is a tensor on the GPU
          initiation_gvf, goal
        )

      logits = self.model(sampled_inputs)
      
      loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(),
        sampled_labels,
        pos_weight=pos_weight,
        weight=weights
      ) 

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      batch_losses.append(loss.item())
    
    return np.mean(batch_losses)

class ConvClassifier(Classifier):
  """"Generic weighted binary convolutional classifier."""
  def __init__(self,
        device,
        threshold=0.5,
        n_input_channels=1,
        batch_size=32,
        lr=1e-3,
        only_reweigh_negative_examples=False):

    self.model = ImageCNN(n_input_channels).to(device)
    super().__init__(device=device, threshold=threshold, batch_size=batch_size, lr=lr,
      only_reweigh_negative_examples=only_reweigh_negative_examples
    )

  def preprocess_batch(self, X):
    assert X.dtype == torch.uint8, X.dtype
    return X.float() / 255.

class MlpClassifier(Classifier):
  """"Generic weighted binary convolutional classifier."""
  def __init__(self,
        device,
        input_dim,
        threshold=0.5,
        batch_size=32,
        lr=1e-3,
        only_reweigh_negative_examples=False):

    self.model = MLP(input_dim).to(device)
    super().__init__(device=device, threshold=threshold, batch_size=batch_size, lr=lr,
      only_reweigh_negative_examples=only_reweigh_negative_examples
    )

  def preprocess_batch(self, X):
    return X

class ClassifierDataset(Dataset):
  def __init__(self, states, labels):
    self.states = states
    self.labels = labels
    super().__init__()

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, i):
    return self.states[i], self.labels[i]
