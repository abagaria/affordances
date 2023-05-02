import ipdb
import torch
import random
import numpy as np
import torch.nn.functional as F

from affordances.utils import utils
from pfrl.nn.atari_cnn import SmallAtariCNN  # TODO: replace with our own copy that works for different image dims
from torch.utils.data import Dataset, DataLoader
from affordances.init_learners.gvf.init_gvf import GoalConditionedInitiationGVF


class ImageCNN(torch.nn.Module):
  def __init__(self, n_input_channels: int = 3, image_dim: int = 84):
    super().__init__()
    if image_dim == 84:
      torso = SmallAtariCNN(n_input_channels=n_input_channels,
                                 n_output_channels=256)
    else:
      assert image_dim == 64, image_dim
      torso = SmallAtariCNN(n_input_channels=n_input_channels,
                                 n_output_channels=256, n_linear_inputs=1152)
    self.model = torch.nn.Sequential(
      torso,
      torch.nn.Linear(in_features=256, out_features=128),
      torch.nn.ReLU(),
      torch.nn.Linear(in_features=128, out_features=1)
    )

  def forward(self, x):
    assert x.max() <= 1.1, (x.max(), x.dtype)
    return self.model(x.float())


class ConvClassifier:
  """"Generic weighted binary convolutional classifier."""
  def __init__(self,
        device,
        only_reweigh_negative_examples: bool,
        threshold=0.5,
        n_input_channels=1,
        batch_size=32,
        lr=1e-3,
        image_dim=64):
    
    self.device = device
    self.is_trained = False
    self.threshold = threshold
    self.batch_size = batch_size
    self.only_reweigh_negative_examples = only_reweigh_negative_examples

    self.model = ImageCNN(n_input_channels, image_dim).to(device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    # Debug variables
    self.losses = []

  @torch.no_grad()
  def predict(self, X, threshold=None):
    logits = self.model(X)
    probabilities = torch.sigmoid(logits)
    threshold = self.threshold if threshold is None else threshold
    return probabilities > threshold

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
    goal: np.ndarray
  ):
    def extract_frame(state):
      if hasattr(state, '_frames'):
        return state._frames[-1]
      return state
    goals = np.repeat(extract_frame(goal), repeats=len(states), axis=0)
    assert isinstance(states, np.ndarray), 'Conversion done in TD(0)'
    assert isinstance(goals, np.ndarray), 'Conversion done in TD(0)'
    assert states.dtype == goals.dtype == np.uint8, 'Preprocessing done in TD(0)'
    values = init_gvf.get_values(states, goals)
    values = utils.tensorfy(values, self.device)  # TODO(ab): keep these on GPU
    weights = values.clip(0., 1.)
    weights[labels == 0] = 1. - weights[labels == 0]

    if self.only_reweigh_negative_examples:
      values[labels == 1] = 1.

    return weights

  def should_train(self, y):
    enough_data = len(y) > self.batch_size
    has_positives = len(y[y == 1]) > 0
    has_negatives = len(y[y != 1]) > 0
    return enough_data and has_positives and has_negatives

  def preprocess_batch(self, X):
    # assert X.dtype == torch.uint8, X.dtype
    return X[:, -1, :, :].unsqueeze(1).float() / 255.

  def fit(self, X, y, initiation_gvf=None, goal=None, n_epochs=1):
    dataset = ClassifierDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

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
      sampled_labels = torch.as_tensor(sampled_labels).to(self.device)
      
      pos_weight = self.determine_pos_weight(sampled_labels)

      # TODO(ab): This is inefficient: we sample many batches before a valid one
      if not pos_weight:
        continue

      if initiation_gvf is not None:
        weights = self.determine_instance_weights(
          np.asarray(sample[0]), # Batch of frame stacks
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
  

def collate_fn(data):
  lazy_frames = [np.asarray(d[0]) for d in data]
  labels = [d[1] for d in data]
  return lazy_frames, labels


class ClassifierDataset(Dataset):
  def __init__(self, states, labels):
    self.states = states
    self.labels = labels
    super().__init__()

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, i):
    return self.states[i], self.labels[i]
