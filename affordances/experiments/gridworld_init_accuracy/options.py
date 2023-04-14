"""Create options for visual gridworld."""


from affordances.agent.hrl.option import Option
from affordances.agent.rainbow.rainbow import Rainbow
from affordances.init_learners.gvf.init_gvf import GoalConditionedInitiationGVF
from affordances.goal_attainment.attainment_classifier import DiscreteInfoAttainmentClassifier


class AgentOverOptions:
  def __init__(
    self,
    env,
    gestation_period,
    timeout,
    image_dim: int,
    gpu: int = 0,
    n_input_channels: int = 1,
    env_steps: int = int(500_000),
    epsilon_decay_steps: int = 25_000,
    final_epsilon: float = 0.1
  ):
    self._env = env
    self._timeout = timeout
    self._gestation_period = gestation_period
    self._gpu = gpu
    self._n_input_channels = n_input_channels

    self.image_dim = image_dim
    
    self.uvfa_policy = self.create_uvfa_policy(
      env_steps, epsilon_decay_steps, final_epsilon)
    
    self.initiation_gvf = self.create_initiation_learner()
    self.subgoals = self.get_subgoals()
    self.options = self.create_options()  # TODO: create global option

  def update_initiation_gvf(self, episode_duration):
    """SGD steps for the initiation GVF."""
    self.initiation_gvf.update(
      n_updates=(episode_duration // self.uvfa_policy.update_interval)
    )

  def create_uvfa_policy(self, env_steps, epsilon_decay_steps, final_eps):
    kwargs = dict(
      n_atoms=51, v_max=10., v_min=-10.,
      noisy_net_sigma=0.5,
      lr=1e-4,
      n_steps=3,
      betasteps=env_steps // 4,
      replay_start_size=1024, 
      replay_buffer_size=int(3e5),
      gpu=self._gpu,
      n_obs_channels=2*self._n_input_channels,
      use_custom_batch_states=False,
      final_epsilon=final_eps,
      epsilon_decay_steps=epsilon_decay_steps,
      image_dim=self.image_dim
    )
    return Rainbow(self._env.action_space.n, **kwargs)
  
  def create_initiation_learner(self):
    """Common goal-conditioned initiation learner shared by all options."""
    return GoalConditionedInitiationGVF(
      target_policy=self.uvfa_policy.agent.batch_act,
      n_actions=self._env.action_space.n,
      n_input_channels=2*self._n_input_channels,
      optimistic_threshold=0.5,
      pessimistic_threshold=0.75,  # don't need this
      image_dim=self.image_dim
    )
  
  def get_subgoal_positions_for_6x6_gridworld(self):
    return [(0, 5), (5, 0), (3, 3), (5, 5)]
  
  def get_subgoal_positions_for_13x13_gridworld(self):
    return [(0,12), (12,0), (6,6), (12,12)]
 
  def get_subgoals(self):
    pos2goals = {}  # pos -> (obs, info)
    subgoal_positions = self.get_subgoal_positions_for_13x13_gridworld()
    for pos in subgoal_positions:
      obs, info = self._env.reset(pos=pos)
      pos2goals[pos] = (obs, info)
    return pos2goals

  def create_options(self) -> list:
    options = []
    goal_attainment_clf = DiscreteInfoAttainmentClassifier("player_pos")
    for i, pos in enumerate(self.subgoals):
      subgoal_obs, subgoal_info = self.subgoals[pos]
      option = Option(i + 1,  # Assuming that none of these are global-option
                      self.uvfa_policy,
                      self.initiation_gvf,
                      goal_attainment_clf,
                      self._gestation_period,
                      self._timeout,
                      exploration_bonus_scale=0,
                      use_her_for_policy_evaluation=True,
                      use_weighted_classifiers=True,
                      subgoal_obs=subgoal_obs,
                      subgoal_info=subgoal_info)
      options.append(option)
    return options
