"""Alternatives for picking which goal to pursue at the curret state."""

import random
import numpy as np

from affordances.utils import utils


def reachability_sample(
    state, info,
    positive_examples,
    parent_classifier,
    gcvf,
    goal_achievement_func,
    sampling_method='argmax'
):
  """Sample the most reachable from the set of feasible goals.
    Args:
      state: np.ndarray current state
      info: info dict corresponding to the current state
      positive_examples: list of lists, each inner list is a (obs, info) traj
      parent_classifier: function from observations -> set membership decisions
      gcvf: goal-conditioned value func maps states, goals -> values
      goal_achievement_func: map from (info, goal_info) -> {0, 1}
      sampling_method: str argmax, softmax, sum
  
    Returns:
      goal_obs: np.ndarray goal observation
      goal_info: dict goal info dictionary
  """
  assert sampling_method in ('argmax', 'softmax', 'sum'), sampling_method

  def get_feasible_goals(goals: list):
    """Convert a list of potential goals to a list of feasible ones."""
    unachieved = _get_unachieved_goals(info, goals, goal_achievement_func)
    goal_observations = np.asarray([goal_tuple[0] for goal_tuple in unachieved])
    predictions = parent_classifier(goal_observations).squeeze(1)
    if predictions.any():
      return [unachieved[i] for i in range(len(unachieved)) if predictions[i]]

  def get_values_for_goals(goals: np.ndarray):
    """Return the values of the current state conditioned on input goals."""
    goal_observations = np.asarray([goal_tuple[0] for goal_tuple in goals])
    states = np.repeat(state[np.newaxis, ...], repeats=len(goals), axis=0)
    return gcvf(states, goal_observations)
  
  def argmax_sample(goals, values):
    return goals[np.argmax(values)]

  def softmax_sample(goals, values, temperature=1.):
    probs = utils.softmax(values, temperature)
    idx = np.random.choice(len(goals), p=probs)
    return goals[idx]

  def sum_sample(goals, values):
    clipped = values.clip(0, 1)
    probs = clipped / clipped.sum()
    idx = np.random.choice(len(goals), p=probs)
    return goals[idx]

  goals = utils.flatten(positive_examples)
  feasible_goals = get_feasible_goals(goals)

  if feasible_goals is not None:
    values = get_values_for_goals(feasible_goals)
    if sampling_method == 'argmax':
      return argmax_sample(feasible_goals, values)
    if sampling_method == 'softmax':
      return softmax_sample(feasible_goals, values)
    if sampling_method == 'sum':
      return sum_sample(feasible_goals, values)
    raise NotImplementedError(sampling_method)


def first_state_sample(
    state, info,
    positive_examples,
    parent_classifier,
    goal_achievement_func
):
  """Sample a random trajectory and pick the first state in term set.
    Args:
      state: np array of the current state
      info: dict of the current state
      positive_examples: list of lists, each inner list is a (obs, info) traj
      parent_classifier: function from observations -> set membership decisions
      goal_achievement_func: if goal is achieved at state
  
    Returns:
      goal_obs: np.ndarray goal observation
      goal_info: dict goal info dictionary
  """
  def get_first_state_in_term_classifier(examples):
    """Given a list of (obs, info) tuples, find the 1st inside the term set."""
    observations = [eg[0] for eg in examples]
    infos = [eg[1] for eg in examples]
    predictions = parent_classifier(observations)
    sampled_idx = predictions.argmax()  # argmax returns the index of 1st max
    return observations[sampled_idx], infos[sampled_idx]

  num_tries = 0

  while num_tries < 100:
    num_tries += 1
    trajectory_idx = random.choice(range(len(positive_examples)))
    sampled_trajectory = positive_examples[trajectory_idx]
    filtered_trajectory = _get_unachieved_goals(
      info, sampled_trajectory, goal_achievement_func)
    if len(filtered_trajectory) > 0:
      state, info = get_first_state_in_term_classifier(filtered_trajectory)
      if parent_classifier([state]):
        return state, info


def _get_unachieved_goals(info, goals, achievement_func):
  """Return subset of goals that are not achieved at the current state.
  Args:
    info: current state
    goals: list of (np.ndarray, dict) tuples representing the candidate goals
    achievement_func: map from (state, goal) -> {0, 1}

  Returns:
    unachieved_goals: list of filtered goals.
  """
  return [goal for goal in goals if not achievement_func(info, goal[1])]
