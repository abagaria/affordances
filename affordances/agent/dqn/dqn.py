import pfrl
import torch
import torch.nn as nn
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN


class SingleSharedBias(nn.Module):
  """Single shared bias used in the Double DQN paper.

  You can add this link after a Linear layer with nobias=True to implement a
  Linear layer with a single shared bias parameter.

  See http://arxiv.org/abs/1509.06461.
  """

  def __init__(self):
    super().__init__()
    self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

  def __call__(self, x):
    return x + self.bias.expand_as(x)
  

def parse_arch(arch, n_actions):
  if arch == "nature":
    return nn.Sequential(
      pnn.SmallAtariCNN(),
      init_chainer_default(nn.Linear(256, n_actions)),
      DiscreteActionValueHead(),
    )
  elif arch == "doubledqn":
    return nn.Sequential(
      pnn.SmallAtariCNN(),
      init_chainer_default(nn.Linear(256, n_actions, bias=False)),
      SingleSharedBias(),
      DiscreteActionValueHead(),
    )
  elif arch == "nips":
    return nn.Sequential(
      pnn.SmallAtariCNN(),
      init_chainer_default(nn.Linear(256, n_actions)),
      DiscreteActionValueHead(),
    )
  elif arch == "dueling":
    return DuelingDQN(n_actions)
  else:
    raise RuntimeError("Not supported architecture: {}".format(arch))


class DQN:
  def __init__(self,
               n_actions: int,
               lr: float,
               epsilon_decay_steps: int,
               prioritized: bool,
               env_steps: int,
               update_interval: int,
               n_step_rewards: int,
               gpu: int
    ):
    q_func = parse_arch("doubledqn", n_actions)
    explorer = explorers.LinearDecayEpsilonGreedy(
      1., 0.05, decay_steps=epsilon_decay_steps
    )
    optimizer = pfrl.optimizers.RMSpropEpsInsideSqrt(
      q_func.parameters(),
      lr=lr,
      alpha=0.95,
      momentum=0.0,
      eps=1e-2,
      centered=True
    )
    
    if prioritized:
      betasteps = env_steps / update_interval
      rbuf = replay_buffers.PrioritizedReplayBuffer(
        10**6,
        alpha=0.6,
        beta=0.4,
        betasteps=betasteps,
        num_steps=n_step_rewards,
      )
    else:
      rbuf = replay_buffers.ReplayBuffer(10**6, n_step_rewards)

    self.agent = agents.DoubleDQN(
      q_func,
      optimizer,
      rbuf,
      gpu=gpu,
      gamma=0.99,
      explorer=explorer,
      replay_start_size=10_000,  # TODO
      target_update_interval=10_000,  # TODO
      update_interval=update_interval,
      batch_accumulator="sum",
      phi=self.phi
    )

  def phi(self, x):
    pass

  def act(self, state):
    pass

  def step(self, state, action, reward, next_state, done, info):
    self._overwrite_pfrl_state(state, action)
    self.agent.observe(next_state, reward, done, )
