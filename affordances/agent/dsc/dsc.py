import random
import numpy as np
from pfrl.replay_buffers import ReplayBuffer
from affordances.agent.dsc.option import Option
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.init_learners.classification.binary_init_classifier import ConvInitiationClassifier


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
    init_learner_type: str,
    goal_info_dict: dict,
    gpu: int = 0,
    n_input_channels: int = 3,
    maintain_init_replay: bool = True,
    max_n_options: int = 10,
    env_steps: int = int(500_000),
    epsilon_decay_steps: int = 25_000,
    exploration_bonus_scale: float = 0,
    optimistic_predict_count_based_bonus:bool = False 
  ):

    self._env = env
    self._timeout = timeout
    self._gestation_period = gestation_period
    self._n_input_channels = n_input_channels
    self._goal_info_dict = goal_info_dict
    self._maintain_init_replay = maintain_init_replay
    self._max_n_options = max_n_options
    self._exploration_bonus_scale = exploration_bonus_scale
    self.optimistic_predict_count_based_bonus = optimistic_predict_count_based_bonus

    self._gpu = gpu
    self._device = f'cuda:{gpu}' if gpu > -1 else 'cpu'
    
    self.task_goal_classifier = task_goal_classifier
    self.start_state_classifier = start_state_classifier
    self.goal_attainment_classifier = goal_attainment_classifier

    # TODO(ab): [refactor] store these inside the start_state_clf object
    self.start_state = start_state
    self.start_state_info = start_info

    self._init_learner_type = init_learner_type
    assert init_learner_type in ('binary', 'td0', 'lstd-rp', 'neural-lstd', 'weighted-binary')

    self.uvfa_policy = self.create_uvfa_policy(
      n_actions=env.action_space.n,
      env_steps=env_steps,
      epsilon_decay_steps=epsilon_decay_steps
    )

    self._init_replay_buffer = None
    if maintain_init_replay:
      self._init_replay_buffer = ReplayBuffer(capacity=int(1e4))

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

    while not done and not reset:
      option = self.select_option(state, info)
      subgoal = option.sample_goal(state)
      
      if subgoal is None:
        subgoal = self.get_subgoal_for_global_option(state)

      state, info, reward, done, reset, reached = option.rollout(
        self._env, state, info, *subgoal, init_replay=self._init_replay_buffer)
      self.manage_chain_after_rollout(option)

      rewards.append(reward)
      print(f"{option} Goal: {subgoal[1]['player_pos']} Reached: {info['player_pos']}")

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
      termination_classifier = parent_option.initiation_learner
    else:
      termination_classifier = self.task_goal_classifier

    classifier = self.create_init_classifier() 
    option =  Option(option_idx=option_idx, uvfa_policy=self.uvfa_policy,
      initiation_learner=classifier,
      parent_initiation_learner=termination_classifier,
      goal_attainment_classifier=self.goal_attainment_classifier,
      gestation_period=self._gestation_period, timeout=self._timeout,
      start_state_classifier=self.start_state_classifier,
      exploration_bonus_scale=self._exploration_bonus_scale, 
      optimistic_predict_count_based_bonus=self.optimistic_predict_count_based_bonus)
    classifier.option = option 
    return option 

  def create_global_option(self):
    return Option(option_idx=0, uvfa_policy=self.uvfa_policy,
      initiation_learner=None,
      parent_initiation_learner=self.task_goal_classifier,
      goal_attainment_classifier=self.goal_attainment_classifier,
      gestation_period=self._gestation_period, timeout=self._timeout // 2,
      start_state_classifier=self.start_state_classifier,
      exploration_bonus_scale=self._exploration_bonus_scale, 
      optimistic_predict_count_based_bonus=self.optimistic_predict_count_based_bonus)

  def create_uvfa_policy(self, n_actions, env_steps, epsilon_decay_steps):
    kwargs = dict(
      n_atoms=51, v_max=10., v_min=-10.,
      noisy_net_sigma=0.5, lr=6.25e-5, n_steps=3,
      betasteps=env_steps // 4,
      replay_start_size=1024, 
      replay_buffer_size=int(3e5),
      gpu=self._gpu, n_obs_channels=2*self._n_input_channels,
      use_custom_batch_states=False,
      final_epsilon=0.1,
      epsilon_decay_steps=epsilon_decay_steps
    )
    return Rainbow(n_actions, **kwargs)

  def create_init_classifier(self):
    if self._init_learner_type == 'binary':
      return ConvInitiationClassifier(self._device,
        optimistic_threshold=0.5,
        pessimistic_threshold=0.75,
        n_input_channels=self._n_input_channels)
    raise NotImplementedError(self._init_learner_type)
