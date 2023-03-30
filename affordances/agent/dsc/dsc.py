from __future__ import annotations

import random
import numpy as np
from affordances.agent.dsc.option import Option
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.init_learners.gvf.init_gvf import GoalConditionedInitiationGVF


class DSCAgent:
  def __init__(self,
    env,
    start_state: np.ndarray,
    start_info: dict,
    start_state_classifier,
    task_goal_classifier,
    goal_attainment_classifier,
    gestation_period: int,
    timeout: int,
    init_gvf_type: str,
    init_classifier_type: str,
    goal_info_dict: dict,
    gpu: int = 0,
    n_input_channels: int = 3,
    maintain_init_replay: bool = True,
    max_n_options: int = 10,
    env_steps: int = int(500_000),
    epsilon_decay_steps: int = 25_000,
    exploration_bonus_scale: float = 0,
    use_her_for_policy_evaluation: bool = False,
    n_actions: int | None = None,
    optimistic_threshold: float = 0.50,
    pessimistic_threshold: float = 0.75,
  ):

    self._env = env
    self._timeout = timeout
    self._gestation_period = gestation_period
    self._n_input_channels = n_input_channels
    self._goal_info_dict = goal_info_dict
    self._maintain_init_replay = maintain_init_replay
    self._max_n_options = max_n_options
    self._exploration_bonus_scale = exploration_bonus_scale
    self._use_her_for_policy_evaluation = use_her_for_policy_evaluation
    self._n_actions = n_actions if n_actions is not None else env.action_space.n
    self._optimistic_threshold = optimistic_threshold
    self._pessimistic_threshold = pessimistic_threshold
    
    self._gpu = gpu
    self._device = f'cuda:{gpu}' if gpu > -1 else 'cpu'
    
    self.task_goal_classifier = task_goal_classifier
    self.start_state_classifier = start_state_classifier
    self.goal_attainment_classifier = goal_attainment_classifier

    # TODO(ab): [refactor] store these inside the start_state_clf object
    self.start_state = start_state
    self.start_state_info = start_info

    self._init_gvf_type = init_gvf_type
    self._init_classifier_type = init_classifier_type
    assert init_gvf_type in ('td0', 'lstd-rp', 'neural-lstd')
    assert init_classifier_type in ('binary', 'weighted-binary')

    self.uvfa_policy = self.create_uvfa_policy(
      env_steps=env_steps,
      epsilon_decay_steps=epsilon_decay_steps
    )

    self.initiation_gvf = self.create_initiation_learner()

    self.chain = []
    self.new_options = []
    self.mature_options = []

    self.goal_option = self.create_new_option()
    self.global_option = self.create_global_option()

    self.chain.append(self.goal_option)
    self.new_options.append(self.goal_option)

  def select_option(self, state, info):
    cond = lambda o, s, i: o.optimistic_is_init_true(s, i) and not o.is_term_true(s, i)
    for option in self.chain:
      if cond(option, state, info):
        return option
    return self.global_option

  def dsc_rollout(self, state, info):
    done = False
    reset = False
    rewards = []
    episode_length = 0

    while not done and not reset:
      option = self.select_option(state, info)
      subgoal = option.sample_goal(state)
      
      if subgoal is None:
        subgoal = self.get_subgoal_for_global_option(state)

      state, info, reward, done, reset, reached, n_steps = option.rollout(
        self._env, state, info, *subgoal)
      self.manage_chain_after_rollout(option)

      rewards.append(reward)
      episode_length += n_steps

      print(f"{option} Goal: {subgoal[1]['player_pos']}",
            f"Reached: {info['player_pos']} Success: {reached}")
    
    self.update_initiation_gvf(episode_length)
      
    discounted_return = (self.uvfa_policy.agent.gamma ** (episode_length - 1)) * sum(rewards)
    estimated_value = self.initiation_gvf.get_values(
      [self.start_state],
      [self.get_subgoal_for_global_option(self.start_state)[0]]
    )
    print(f"Discounted return: {discounted_return}",
          f"Estimated value: {estimated_value}")

    return state, info, rewards

  def get_subgoal_for_global_option(self, state):
    """Use goal samples if possible, else use the 0 tensor."""
    samples = list(self.global_option.effect_set) + \
              list(self.goal_option.effect_set)
    if samples:
      return random.choice(samples)
    return np.zeros_like(state), self._goal_info_dict

  def manage_chain_after_rollout(self, option: Option):
    """Chain maintainence: gestate/create new options."""

    if option in self.new_options and option.training_phase != 'gestation':
      self.new_options.remove(option)
      self.mature_options.append(option)

      if option.training_phase != 'gestation' and not option._is_global_option \
        and option.should_expand_initiation(self.start_state, self.start_state_info):
        print(f'Expanding the skill chain to fix {option} to s0')
        option.is_last_option = True
    
    if self.should_create_new_option():
      new_option = self.create_new_option()
      self.chain.append(new_option)
      self.new_options.append(new_option)
  
  def update_initiation_gvf(self, episode_duration):
    if len(self.goal_option.initiation_classifier.positive_examples) > 0:
      self.initiation_gvf.update(
        n_updates=(episode_duration // self.uvfa_policy.update_interval)
      )

  def should_create_new_option(self):
    """Create a new option if the following conditions are satisfied:
      - we are not currently learning any option (they are all mature),
      - the start state of the MDP is not inside the init-set of some option.
    """
    return len(self.mature_options) > 0 and \
      len(self.new_options) == 0 and \
      len(self.mature_options) < self._max_n_options and \
      self.mature_options[-1].training_phase != 'gestation' and \
        not self.contains_init_state()

  def contains_init_state(self):
    """Whether the start state inside any option's initiation set."""
    for option in self.mature_options:
      if option.optimistic_is_init_true(self.start_state, self.start_state_info):
        return True
    return False

  def create_new_option(self):
    option_idx = len(self.chain) + 1
    
    if option_idx > 1:
      parent_option = self.chain[-1]
      assert isinstance(parent_option, Option)
      termination_classifier = parent_option.pessimistic_is_init_true
      parent_initiation_classifier = parent_option.initiation_classifier
    else:
      termination_classifier = self.task_goal_classifier
      parent_initiation_classifier = self.task_goal_classifier

    return Option(option_idx=option_idx, uvfa_policy=self.uvfa_policy,
      initiation_gvf=self.initiation_gvf,
      termination_classifier=termination_classifier,
      goal_attainment_classifier=self.goal_attainment_classifier,
      gestation_period=self._gestation_period, timeout=self._timeout,
      start_state_classifier=self.start_state_classifier,
      exploration_bonus_scale=self._exploration_bonus_scale,
      use_her_for_policy_evaluation=self._use_her_for_policy_evaluation,
      parent_initiation_classifier=parent_initiation_classifier,
      use_weighted_classifiers=self._init_classifier_type=="weighted-binary")

  def create_global_option(self):
    return Option(option_idx=0, uvfa_policy=self.uvfa_policy,
      initiation_gvf=self.initiation_gvf,
      termination_classifier=self.task_goal_classifier,
      goal_attainment_classifier=self.goal_attainment_classifier,
      gestation_period=self._gestation_period, timeout=self._timeout // 2,
      start_state_classifier=self.start_state_classifier,
      exploration_bonus_scale=self._exploration_bonus_scale,
      use_her_for_policy_evaluation=self._use_her_for_policy_evaluation,
      parent_initiation_classifier=self.task_goal_classifier,
      use_weighted_classifiers=self._init_classifier_type=="weighted-binary")

  def create_uvfa_policy(self, env_steps, epsilon_decay_steps):
    kwargs = dict(
      n_atoms=51, v_max=10., v_min=-10.,
      noisy_net_sigma=0.5,
      lr=1e-4,
      n_steps=3,
      betasteps=env_steps // 4,
      replay_start_size=1024, 
      replay_buffer_size=int(3e5),
      gpu=self._gpu, n_obs_channels=2*self._n_input_channels,
      use_custom_batch_states=False,
      final_epsilon=0.1,
      epsilon_decay_steps=epsilon_decay_steps
    )
    return Rainbow(self._n_actions, **kwargs)

  def create_initiation_learner(self):
    """Common goal-conditioned initiation learner shared by all options."""
    if self._init_gvf_type == 'td0':
      return GoalConditionedInitiationGVF(
        target_policy=self.uvfa_policy.agent.batch_act,
        n_actions=self._n_actions,
        n_input_channels=2*self._n_input_channels,
        optimistic_threshold=self._optimistic_threshold,
        pessimistic_threshold=self._pessimistic_threshold
      )
    raise NotImplementedError()
