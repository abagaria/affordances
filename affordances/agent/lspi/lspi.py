"""Wrapper class for LSPI."""

import jax
import numpy as np
import jax.numpy as jnp

from . import lspi_lib


class LSPI:
  """Least Squares Policy Iteration."""
  def __init__(self,
      states, actions, rewards, next_states, dones,
      n_actions, n_state_features, extraction_method, seed=42):
    """Constructor for LSPI.

    Args:
      states: state part of (s, a, r, s', d)
      actions: action part of (s, a, r, s', d)
      rewards: reward part of (s, a, r, s', d). For many-goals version,
      it should be a matrix where each row is a different goal-conditioned
      reward vector
      next_states: s' part of (s, a, r, s', d)
      dones: terminal signal for each transitions. Same logic as rewards for
      many-goals version of the done matrix
      n_actions: number of discrete actions in the env
      n_state_features: number of features per state
      extraction_method: string identity/random projections
      seed: int random seed for all JAX functions
    """
    assert extraction_method in ('identity', 'random')
    self._extraction_method = extraction_method

    self.states = jnp.array(states)
    self.actions = jnp.array(actions)
    self.rewards = jnp.array(rewards)
    self.next_states = jnp.array(next_states)
    self.dones = jnp.array(dones)

    self._seed = jax.random.PRNGKey(seed)
    self._many_goals_version = self.rewards.ndim == 2 and self.rewards.shape[0] > 1
    assert self.rewards.shape == self.dones.shape

    self.n_actions = n_actions
    self.n_state_features = n_state_features
    
    self._projection = jax.random.normal(
      self._seed, (self.n_state_features, self.n_features)
    ) if extraction_method == 'random' else jnp.eye(n_state_features)

    self._extractor = lspi_lib.random_feature_extractor
    self._lspi_fn = lspi_lib.batched_lspi if self._many_goals_version else lspi_lib.lspi

    self.state_action_matrix = self.construct_state_action_matrix(self.next_states)
    print(self.n_features, self.state_action_matrix.shape, n_actions)

  @property
  def n_features(self):
    """Number of features per state."""
    if self._extraction_method == 'identity':
      return self.n_state_features
    return 100
    # return int(np.sqrt(len(self.states)))  # Based on the LSPI-RP paper
  
  def construct_state_action_matrix(self, states):
    repeated_states = jnp.repeat(states, self.n_actions, axis=0)
    actions = jnp.arange(self.n_actions)[jnp.newaxis, :]
    repeated_actions = jnp.repeat(actions, len(states), axis=0).reshape(-1)
    return self._extractor(repeated_states, repeated_actions,
      self.n_features, self.n_actions, self._projection)

  def __call__(self):
    return self._lspi_fn(
        self._seed, self.states, self.actions, self.rewards,
      self.next_states, self.dones, self.state_action_matrix, self.n_features,
      self.n_actions, self._projection
    )
  
  def _get_q_values(self, theta, states=None):
    sa_matrix = self.state_action_matrix if states is None \
      else self.construct_state_action_matrix(states)
    if self._many_goals_version:
      n_goals = theta.shape[0]
      n_states = len(states) if states is not None else len(self.next_states)
      return (sa_matrix @ theta.T).reshape(n_states, -1, n_goals)
    return (sa_matrix @ theta).reshape(-1, self.n_actions)
  
  def get_values(self, theta, states=None):
    """Get values for states. If no states are given, use the original ones."""
    q_values = self._get_q_values(theta, states)
    return np.max(q_values, axis=1)

  def get_seek_avoid_value_functions(self, theta, done_matrix):
    assert self._many_goals_version, 'need seek/avoid rewards first'
    
    n_goals = theta.shape[0] // 2
    print(f'Getting seek/avoid vfs for {n_goals} goals')
    
    values = self.get_values(theta)
    seek_vf = np.array(values)[:, :n_goals]
    avoid_vf = np.array(values)[:, n_goals:]

    # Set the values at terminal states
    seek_dones = done_matrix.T[:, :n_goals]
    seek_vf[seek_dones] = 1.
    avoid_vf[seek_dones] = -1.

    return seek_vf, avoid_vf
