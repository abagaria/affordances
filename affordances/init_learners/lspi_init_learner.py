import time
import torch
import numpy as np
import torch.nn.functional as F

from pfrl.nn.mlp import MLP
from pfrl.nn.atari_cnn import SmallAtariCNN
from pfrl.replay_buffers import ReplayBuffer
from affordances.agent.lspi.lspi import LSPI
from affordances.utils import utils


class LearnerFunc(torch.nn.Module):
  """Function that maps obs -> initiation prob/value."""
  def __init__(self) -> None:
      super().__init__()
      self.model = torch.nn.Sequential(
        SmallAtariCNN(
          n_input_channels=3,
          n_output_channels=256),
        MLP(in_size=256, out_size=1, hidden_sizes=(128, 128))
      )

  def forward(self, obs):
    return self.model(obs)


class LSPIInitLearner:
  """Model that regresses state to Init value using LSPI as GT."""

  def __init__(
    self,
    n_actions: int,
    replay_size: int = int(1e5),
    batch_size: int = 32,
    device: str = 'cuda',
    lr: float = 1e-3
  ):
      self._n_actions = n_actions
      self._batch_size = batch_size
      self.replay = ReplayBuffer(capacity=replay_size)
      self.device = torch.device(device)

      self.init_learner = LearnerFunc().to(self.device)
      
      self._optimizer = torch.optim.Adam(
        self.init_learner.parameters(), lr=lr)


  @torch.no_grad()
  def __call__(self, states) -> np.ndarray:
    states = utils.tensorfy(states, device=self.device)
    values = self.init_learner(states)
    return values.cpu().numpy()

  def get_values(self, params, states):
    del params
    return self(states)

  def add_transition(self, obs, action, reward, next_obs, info):
    """Add transition (s, a, r, s', gamma) to the replay buffer."""
    self.replay.append(obs, action, reward, next_obs,
    is_state_terminal=info['terminated'], extra_info=info)

  def update(self):
    """Sample a batch of transitions & update the params."""
    if len(self.replay) >= self._batch_size:
      states, gt_values = self._get_labeled_data()

      state_tensor = utils.tensorfy(
        states,
        device=self.device,
        transpose_to=(0, 3, 1, 2))

      predicted_values = self.init_learner(state_tensor).squeeze()

      loss = F.mse_loss(predicted_values, gt_values)

      self._optimizer.zero_grad()
      loss.backward()
      self._optimizer.step()

  def _create_init_learner(self, transitions):
    n_obs_features = 84 * 84 * 3  # number of features (pixels) in raw obs
    return LSPI(*transitions,
      n_actions=self._n_actions,
      n_state_features=n_obs_features,
      extraction_method='random')

  @torch.no_grad()
  def _get_labeled_data(self):
    """Use LSPI to get the regression labels for value learning."""
    t0 = time.time()

    # Sample and prepare transitions
    n_samples = min(len(self.replay), self._batch_size)

    sampled_n_step_transitions = self.replay.sample(n_samples)

    # Prepare transitions for LSPI
    sampled_transitions = [n_step_transitions[-1] for n_step_transitions in sampled_n_step_transitions]
    parsed_transitions = utils.unpack_transitions(sampled_transitions)
    
    # Fit a value function \hat{V^*}
    lspi_agent = self._create_init_learner(parsed_transitions)
    params = lspi_agent()

    lspi_values = utils.tensorfy(
        np.array(lspi_agent.get_values(params)), device=self.device)

    # print(f'Took {time.time() - t0}s to get LSPI values')
    return parsed_transitions[0], lspi_values
