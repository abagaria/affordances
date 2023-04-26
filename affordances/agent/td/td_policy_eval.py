"""Policy evaluation using TD(0)."""

import ipdb
import copy
import pfrl
import torch
import numpy as np
import torch.nn.functional as F

from torch.optim import Adam
from pfrl.replay_buffer import batch_experiences
from pfrl.utils.copy_param import synchronize_parameters
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer

from affordances.utils.nn import SmallAtariCNN


class QNetwork(torch.nn.Module):
  def __init__(self, n_input_channels, n_actions, image_dim=84, use_sigmoid=False):
    super().__init__()
    self._use_sigmoid = use_sigmoid

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
      torch.nn.Linear(in_features=128, out_features=n_actions)
    )

  def forward(self, x):
    if self._use_sigmoid:
      return torch.sigmoid(self.model(x.float()))
    return self.model(x.float())


class TDPolicyEvaluator:
  """Policy evaluation using 1-step temporal difference errors."""

  def __init__(
    self,
    replay: pfrl.replay_buffers.ReplayBuffer,
    n_actions: int,
    n_input_channels: int,
    tau: float = 5e-3,
    batch_size: int = 32,
    gamma: float = 0.99,
    learning_rate: float = 1e-4,
    image_dim : int = 84
  ):
    """Constructor for TD policy evaluation.

    Args:
      replay: pointer to the agent's replay buffer.
      n_actions (int): number of discrete actions in the env.
      n_input_channels (int): e.g, 1 for grayscale, 3 for color, etc
      tau: (float) soft update param for updating target net.
      batch_size (int)
      gamma (float): discount factor
      learning_rate (float): lr for optimizer
    """

    self._tau = tau
    self._gamma = gamma
    self._replay = replay
    self._n_actions = n_actions
    self._batch_size = batch_size
    self._learning_rate = learning_rate
    self._maintain_prioritized_buffer = isinstance(replay, PrioritizedReplayBuffer)

    self._online_q_network = QNetwork(n_input_channels, n_actions, image_dim, use_sigmoid=True)
    self._target_q_network = copy.deepcopy(self._online_q_network).eval().requires_grad_(False)

    self._n_updates = 0
    self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    self._online_q_network.to(self._device)
    self._target_q_network.to(self._device)
    self._optimizer = Adam(self._online_q_network.parameters(), lr=learning_rate)

    print(f'Constructed TD(0) policy evaluator with prioritization={self._maintain_prioritized_buffer}')

  @staticmethod
  def phi(x):
    """ Observation pre-processing for convolutional layers. """
    return np.asarray(x, dtype=np.float32) / 255.

  @torch.no_grad()
  def get_values(self, states):
    assert isinstance(states, np.ndarray), type(states)
    assert states.dtype == np.uint8, states.dtype
    qvalues = self._online_q_network(
      torch.as_tensor(states, device=self._device).float() / 255.
    )
    return torch.max(qvalues, dim=1).values.cpu().numpy()

  def train(self, target_policy):
    """Single minibatch update of the evaluation value-function."""
    transitions = batch_experiences(
      self._replay.sample(self._batch_size),
      self._device,
      self.phi,
      self._gamma
    )

    states = transitions['state']
    actions = transitions['action']
    rewards = transitions['reward']
    next_states = transitions['next_state']
    dones = transitions['is_state_terminal']
    
    predicted_qvalues = self._online_q_network(states)
    prediction = predicted_qvalues[range(len(states)), actions]

    with torch.no_grad():
      discount = (1. - dones) * self._gamma
      next_actions = target_policy(next_states)
      next_qvalues = self._target_q_network(next_states)
      next_values = next_qvalues[range(len(next_states)), next_actions]
      target = rewards + (discount * next_values)

      if self._maintain_prioritized_buffer:
        td_errors = torch.abs(target - prediction)
        self._replay.update_errors(td_errors.detach().cpu().numpy())

    loss = F.mse_loss(prediction, target)

    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

    self._update_target_network()
    # print(f'TDEvaluator Step#{self._n_updates} Loss: {loss.item()}')
    self._n_updates += 1

  @torch.no_grad()
  def _update_target_network(self):
    synchronize_parameters(
      self._online_q_network, self._target_q_network, 'soft', tau=self._tau)
