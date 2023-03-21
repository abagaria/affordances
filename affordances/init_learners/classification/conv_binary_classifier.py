import torch
import random
import numpy as np
import torch.nn.functional as F

from affordances.utils import utils
from pfrl.nn.atari_cnn import SmallAtariCNN
from torch.utils.data import Dataset, DataLoader


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
    self.model = torch.nn.Sequential(
      torch.nn.Linear(in_features=input_dim, out_features=256)
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
        batch_size=32):
    
    self.device = device
    self.is_trained = False
    self.threshold = threshold
    self.batch_size = batch_size

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

  def should_train(self, y):
    enough_data = len(y) > self.batch_size
    has_positives = len(y[y == 1]) > 0
    has_negatives = len(y[y != 1]) > 0
    return enough_data and has_positives and has_negatives

  def preprocess_batch(self, X):
    raise NotImplementedError

  def fit(self, X, y, n_epochs=1):
    dataset = ClassifierDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    if self.should_train(y):
      losses = []

      for _ in range(n_epochs):
        epoch_loss = self._train(dataloader)                
        losses.append(epoch_loss)
      
      self.is_trained = True

      mean_loss = np.mean(losses)
      self.losses.append(mean_loss)

  def _train(self, loader):
    """ Single epoch of training. """
    batch_losses = []

    for sampled_inputs, sampled_labels in loader:
      sampled_inputs = utils.tensorfy(sampled_inputs, self.device)
      sampled_inputs = self.preprocess_batch(sampled_inputs)
      sampled_labels = sampled_labels.to(self.device)
      
      pos_weight = self.determine_pos_weight(sampled_labels)

      # TODO(ab): This is inefficient: we sample many batches before a valid one
      if not pos_weight:
        continue

      logits = self.model(sampled_inputs)
      
      loss = F.binary_cross_entropy_with_logits(
        logits.squeeze(),
        sampled_labels,
        pos_weight=pos_weight
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
        lr=1e-3):

    super().__init__(device=device, threshold=threshold, batch_size=batch_size)
    self.model = ImageCNN(n_input_channels).to(device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        lr=1e-3):

    super().__init__(device=device, threshold=threshold, batch_size=batch_size)
    self.model = MLP(input_dim).to(device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
