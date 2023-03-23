import ipdb
import random
import numpy as np
import collections

from affordances.agent.rainbow.rainbow import Rainbow
from affordances.init_learners.gvf.init_gvf import GoalConditionedInitiationGVF
from affordances.goal_attainment.attainment_classifier import GoalAttainmentClassifier


class Option:
  def __init__(self,
    option_idx: int,
    uvfa_policy: Rainbow,
    initiation_learner: GoalConditionedInitiationGVF,
    termination_classifier: GoalConditionedInitiationGVF,
    parent_positive_examples: list,
    goal_attainment_classifier: GoalAttainmentClassifier,
    gestation_period: int,
    timeout: int,
    start_state_classifier,
    exploration_bonus_scale: float,
    use_her_for_policy_evaluation: bool):
      self._timeout = timeout
      self._solver = uvfa_policy
      self._option_idx = option_idx
      self._gestation_period = gestation_period
      self._goal_attainment_classifier = goal_attainment_classifier
      self._exploration_bonus_scale = exploration_bonus_scale
      self._use_her_for_policy_evaluation = use_her_for_policy_evaluation
      
      self.initiation_learner = initiation_learner
      self.termination_classifier = termination_classifier
      self.parent_positive_examples = parent_positive_examples
      
      # function that maps current info to {0, 1}
      self._start_state_classifier = start_state_classifier

      self.is_last_option = False
      self.expansion_classifier = None

      self._is_global_option = option_idx == 0
      
      self.num_goal_hits = 0
      self.num_executions = 0
      self.effect_set = collections.deque(maxlen=20)

      self.positive_examples = collections.deque([], maxlen=10)
      self.pessimistic_initiation_samples = []

      # Information for logging and debugging
      self.debug_log = collections.defaultdict(list)

      print(f'Created {self} with bonus_scale={exploration_bonus_scale}')

  @property
  def training_phase(self):
    return 'gestation' if self.num_goal_hits < self._gestation_period else 'mature'
  
  def optimistic_is_init_true(self, state, info):
    if self._is_global_option or self.training_phase == 'gestation':
      return True

    if self.is_last_option and self._start_state_classifier(info):
      return True
    
    goals = self.get_potential_goals(state, info)

    if goals:
      goals = np.asarray([goal[0] for goal in goals])
      states = np.repeat(state[np.newaxis, ...], repeats=len(goals), axis=0)

      decision1 = self.initiation_learner.optimistic_predict(states, goals) 
      decision2 = self.initiation_learner.pessimistic_predict(states, goals)
      return decision1 or decision2
    
    return False

  def pessimistic_is_init_true(self, state, info):
    goals = self.get_potential_goals(state, info)
    
    if goals:
      goals = np.asarray([goal[0] for goal in goals])
      states = np.repeat(state[np.newaxis, ...], repeats=len(goals), axis=0)

      return self.initiation_learner.pessimistic_predict(states, goals)
    
    return False

  def is_term_true(self, state, info):
    if self._option_idx <= 1:
      return self.termination_classifier(info)
    
    # TODO(ab): maybe task goal should be in every option's termination set
    return info['terminated'] or self.termination_classifier(state, info)

  def at_local_goal(self, state, info, goal, goal_info):
    inputs = (state, goal) if self._goal_attainment_classifier.use_obs else (info, goal_info)
    reached_goal = self._goal_attainment_classifier(*inputs)
    return reached_goal# and self.is_term_true(state, info)

  def act(self, state, goal):
    augmented_state = self.get_augmeted_state(state, goal)
    return self._solver.act(augmented_state)

  def sample_goal(self, state, info):
    """Sample a goal to pursue from the option's termination region."""

    if self._option_idx > 1:
      goals = self.get_potential_goals(state, info) 

      if len(goals) > 0:
        
        values = self.get_value_for_all_goals(state, goals)
        idx = np.argmax(values)
        print(
          f'{self} s={info["player_pos"]}',
          f'g={goals[idx][1]["player_pos"]}',
          f'v(s,g)={values[idx]}'
        )
        return goals[idx]
      
    elif len(self.effect_set) > 0:
      return random.choice(self.effect_set)
      
  def get_potential_goals(self, state, info):
    """Return the set of parent positives that are not achieved in s_t."""
    def get_all_goals():
      """Return all parent positives."""
      if self._option_idx <= 1:
        return list(self.effect_set)
      return self.parent_positive_examples
    gcrf = self._goal_attainment_classifier
    return [g for g in get_all_goals() if not gcrf(info, g[1])]
    # return [goal for goal in unachieved_goals if self.is_term_true(*goal)]
  
  def get_value_for_all_goals(self, state, goals):
    goals = np.asarray([example[0] for example in goals])
    states = np.repeat(state[np.newaxis, ...], repeats=len(goals), axis=0)
    values = self.initiation_learner.get_values(states, goals)
    return values
  
  def add_trajectory_to_positive_examples(self, transitions):
    infos = [transition[-1] for transition in transitions]
    observations = [transition[-2] for transition in transitions]

    examples = [(obs, info) for obs, info in zip(observations, infos)]
    self.positive_examples.append(examples)

  def rollout(self, env, state, info, goal, goal_info, init_replay):
    """Execute the option policy from `state` towards `goal`."""
       
    done = False
    reset = False
    reached = False

    n_steps = 0
    total_reward = 0.
    trajectory = []  # (s, a, r, s', info)

    while not done and not reset and not reached and n_steps < self._timeout:
      action = self.act(state, goal)
      
      next_state, reward, done, info = env.step(action)

      n_steps += 1
      trajectory.append((state, action, reward, next_state, info))
  
      if init_replay is not None:
        init_replay.append(state, action, reward, next_state,
          is_state_terminal=info['terminated'], extra_info=info)
  
      reached = self.at_local_goal(next_state, info, goal, goal_info)

      state = next_state
      reset = info['needs_reset']
      total_reward += reward

    self.update_option_after_rollout(goal, goal_info, trajectory, reached)
    self.log_progress(info, reached, goal_info)

    return state, info, total_reward, done, reset, reached, n_steps
  
  def update_option_after_rollout(self, goal, goal_info, transitions, success):
    if success:
      self.num_goal_hits += 1

      final_state = transitions[-1][-2]
      final_info = transitions[-1][-1]
      self.effect_set.append((final_state, final_info))

      if not self._is_global_option:
        self.add_trajectory_to_positive_examples(transitions)

    self.update_option_params(transitions, goal, goal_info)

  def update_option_params(self, transitions, goal, goal_info):
    """Update the parameters of the UVFA and the initiation GVF."""

    self._solver.experience_replay(
      self.relabel_trajectory(transitions, goal, goal_info, add_bonus=True))
    self.initiation_learner.add_trajectory_to_replay(
      self.relabel_trajectory(transitions, goal, goal_info, add_bonus=False))

    hindsight_goal, hindsight_goal_info = self.pick_hindsight_goal(transitions)

    # If the first state in the trajectory doesn't achieve the hindsight goal
    if not self._goal_attainment_classifier(transitions[0][-1], hindsight_goal_info):
      self._solver.experience_replay(
        self.relabel_trajectory(
        transitions, hindsight_goal, hindsight_goal_info, add_bonus=True))
      
      if self._use_her_for_policy_evaluation:
        self.initiation_learner.add_trajectory_to_replay(
          self.relabel_trajectory(
          transitions, hindsight_goal, hindsight_goal_info, add_bonus=False))

  def relabel_trajectory(self, transitions, goal, goal_info, add_bonus=True):
    relabeled_trajectory = []
    bonus_scale = self._exploration_bonus_scale if add_bonus else 0
    for state, action, _, next_state, info in transitions:
      sg = self.get_augmeted_state(state, goal)
      nsg = self.get_augmeted_state(next_state, goal)
      reached = self._goal_attainment_classifier(info, goal_info)
      reward = float(reached) + (bonus_scale * info['bonus'])
      relabeled_trajectory.append((sg, action, reward, nsg, reached, info))
      if reached:
        break
    return relabeled_trajectory

  def pick_hindsight_goal(self, transitions, method='final'):
    if method == 'final':
      goal_transition = transitions[-1]
      goal = goal_transition[-2]
      goal_info = goal_transition[-1]
      return goal, goal_info
    if method == 'future':
      start_idx = len(transitions) // 2
      goal_idx = random.randint(start_idx, len(transitions) - 1)
      goal_transition = transitions[goal_idx]
      goal = goal_transition[-2]
      goal_info = goal_transition[-1]
      return goal, goal_info
    raise NotImplementedError(method)

  def should_expand_initiation(self, s0: np.ndarray, info0: dict):
    """Check if the option can initiate at the start-state."""
    if self.training_phase != 'gestation':
      return self.pessimistic_is_init_true(s0, info0)
    return False
  
  def get_augmeted_state(self, state, goal):
    assert state.shape == (1, 84, 84), state.shape
    return np.concatenate((state, goal), axis=0)
  
  def log_progress(self, info, success, goal_info):
    self.debug_log['timesteps'].append(info['timestep'])
    self.debug_log['successes'].append(success)
    self.debug_log['goals'].append(goal_info)
    self.debug_log['phases'].append(self.training_phase)

  def __hash__(self) -> int:
    return self._option_idx

  def __str__(self) -> str:
    return f'option-{self._option_idx}'
  
  def __repr__(self) -> str:
    return str(self)
