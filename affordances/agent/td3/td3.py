import numpy as np

import torch
from torch import nn

import pfrl
from pfrl import explorers, replay_buffers, utils

class TD3:

  # https://github.com/pfnet/pfrl/blob/master/examples/mujoco/reproduction/td3/train_td3.py
  def __init__(self, action_space, obs_space, replay_start_size, gpu, sigma, lr, batch_size):
    
    action_size = action_space.low.size
    obs_size = obs_space.low.size

    policy = nn.Sequential(
        nn.Linear(obs_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, action_size),
        nn.Tanh(),
        pfrl.policies.DeterministicHead(),
    )
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr)

    def make_q_func_with_optimizer():
        q_func = nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr)
        return q_func, q_func_optimizer

    q_func1, q_func1_optimizer = make_q_func_with_optimizer()
    q_func2, q_func2_optimizer = make_q_func_with_optimizer()

    rbuf = replay_buffers.ReplayBuffer(10**6)

    explorer = explorers.AdditiveGaussian(
        scale=sigma, low=action_space.low, high=action_space.high
    )

    def burnin_action_func():
        """Select random actions until model is updated one or more times."""
        return np.random.uniform(action_space.low, action_space.high).astype(np.float32)

    # Hyperparameters in http://arxiv.org/abs/1802.09477
    self.agent = pfrl.agents.TD3(
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        rbuf,
        gamma=0.99,
        soft_update_tau=5e-3,
        explorer=explorer,
        replay_start_size=replay_start_size,
        gpu=gpu,
        minibatch_size=batch_size,
        burnin_action_func=burnin_action_func,
    )

  def act(self, obs):
    return self.agent.act(obs)

  def _overwrite_pfrl_state(self, state, action):
    """ Hack the pfrl state so that we can call act() consecutively during an episode before calling step(). """
    self.agent.batch_last_obs = [state]
    self.agent.batch_last_action = [action] 

  def step(self, state, action, reward, next_state, done, reset):
    """ Learning update based on a given transition from the environment. """
    # self._overwrite_pfrl_state(state, action)
    self.agent.observe(next_state, reward, done, reset)