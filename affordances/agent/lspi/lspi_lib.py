"""Goal-conditioned LSPI implemented in pure JAX."""


import jax
import jax.numpy as jnp


def state_action_features(state_features, action, n_features, n_actions):
  """Copy-paste technique for creating s-a features (as used in LSPI paper)."""
  start_idx = n_features * action
  features = jnp.zeros((n_features * n_actions,), dtype=state_features.dtype)
  features = jax.lax.dynamic_update_slice(features, state_features, (start_idx,))
  return features


batched_state_action_features = jax.vmap(
  state_action_features,
  in_axes=(0, 0, None, None)
)


def random_feature_extractor(states, actions, n_dims, n_actions, proj):
  """Random projections for (s, a) feature extraction."""
  state_features = states.reshape((len(states), -1))
  projected_states = state_features @ proj
  return batched_state_action_features(
    projected_states, actions, n_dims, n_actions
  )


def initialize_policy(seed, n_states, n_actions):
  return jax.random.choice(seed, n_actions, shape=(n_states,))


def select_actions(theta, phi_matrix, n_actions):
  """Policy improvement: given the Q-function, select actions at s'."""
  values = phi_matrix @ theta
  q_values = values.reshape(-1, n_actions)
  policy = jnp.argmax(q_values, axis=1)
  return policy


def lstdq(next_actions,
      states, actions, rewards, next_states, dones,
      projection, n_actions, gamma, n_dims):
  """Policy Evaluation: Least Squares TD for Q-function with Random Projections."""
  phi = random_feature_extractor(states, actions, n_dims, n_actions, projection)
  next_phi = random_feature_extractor(next_states, next_actions, n_dims, n_actions, projection)
  discount = (1. - dones) * gamma
  discount_matrix = jnp.repeat(discount[:, jnp.newaxis], n_dims * n_actions, axis=1)
  discounted_next_phi = discount_matrix * next_phi
  a = phi.T @ (phi - discounted_next_phi)
  b = phi.T @ rewards
  return jnp.linalg.lstsq(a, b)[0]


def lspi(seed, states, actions, rewards, next_states, dones,
     state_action_matrix, n_dims, n_actions, projection, gamma=0.95):
  """Least Squares Policy Iteration with Random Projections."""
  next_actions = initialize_policy(seed, len(states), n_actions)
  theta = lstdq(next_actions, states, actions, rewards, next_states, dones,
    projection, n_actions, gamma, n_dims)
  next_actions = select_actions(theta, state_action_matrix, n_actions)
  return lstdq(next_actions, states, actions, rewards, next_states, dones,
    projection, n_actions, gamma, n_dims)


batched_lspi = jax.vmap(
  lspi,
  in_axes=(None, None, None, 0, None, 0, None, None, None, None)
)
