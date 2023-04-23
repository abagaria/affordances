import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from pfrl.utils import batch_states
from pfrl.nn.atari_cnn import constant_bias_initializer
from pfrl.initializers import init_chainer_default


def batch_experiences(experiences, device, phi, gamma, batch_states=batch_states):
    """Takes a batch of k experiences each of which contains j

    consecutive transitions and vectorizes them, where j is between 1 and n.

    Args:
        experiences: list of experiences. Each experience is a list
            containing between 1 and n dicts containing
              - state (object): State
              - action (object): Action
              - reward (float): Reward
              - is_state_terminal (bool): True iff next state is terminal
              - next_state (object): Next state
        device : GPU or CPU the tensor should be placed on
        phi : Preprocessing function
        gamma: discount factor
        batch_states: function that converts a list to a batch
    Returns:
        dict of batched transitions
    """

    batch_exp = {
        "state": batch_states([elem[0]["state"] for elem in experiences], device, phi),
        "action": torch.as_tensor(
            [elem[0]["action"] for elem in experiences], device=device
        ),
        "reward": torch.as_tensor(
            [
                sum((gamma ** i) * exp[i]["reward"] for i in range(len(exp)))
                for exp in experiences
            ],
            dtype=torch.float32,
            device=device,
        ),
        "next_state": batch_states(
            [elem[-1]["next_state"] for elem in experiences], device, phi
        ),
        "is_state_terminal": torch.as_tensor(
            [
                any(transition["is_state_terminal"] for transition in exp)
                for exp in experiences
            ],
            dtype=torch.float32,
            device=device,
        ),
        "discount": torch.as_tensor(
            [(gamma ** len(elem)) for elem in experiences],
            dtype=torch.float32,
            device=device,
        ),
        "extra_info": [elem[-1]["extra_info"] for elem in experiences]
    }
    if all(elem[-1]["next_action"] is not None for elem in experiences):
        batch_exp["next_action"] = torch.as_tensor(
            [elem[-1]["next_action"] for elem in experiences], device=device
        )
    return batch_exp


class RandomizeAction(gym.ActionWrapper):
    """Apply a random action instead of the one sent by the agent.

    This wrapper can be used to make a stochastic env. The common use is
    for evaluation in Atari environments, where actions are replaced with
    random ones with a low probability.

    Only gym.spaces.Discrete is supported as an action space.

    For exploration during training, use explorers like
    pfrl.explorers.ConstantEpsilonGreedy instead of this wrapper.

    Args:
        env (gym.Env): Env to wrap.
        random_fraction (float): Fraction of actions that will be replaced
            with a random action. It must be in [0, 1].
    """

    def __init__(self, env, random_fraction):
        super().__init__(env)
        assert 0 <= random_fraction <= 1
        assert isinstance(
            env.action_space, gym.spaces.Discrete
        ), "RandomizeAction supports only gym.spaces.Discrete as an action space"
        self._random_fraction = random_fraction
        self._np_random = np.random.RandomState()

    def action(self, action):
        if self._np_random.rand() < self._random_fraction:
            return self.action_space.sample()
        else:
            return action

    def seed(self, seed):
        super().seed(seed)
        self._np_random.seed(seed)


class SmallAtariCNN(torch.nn.Module):
    """Small CNN module proposed for DQN in NeurIPS DL Workshop, 2013.

    See: https://arxiv.org/abs/1312.5602
    """

    def __init__(
        self, n_input_channels=4, n_output_channels=256,
        activation=F.relu, bias=0.1, n_linear_inputs=2592
    ):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 16, 8, stride=4),
                nn.Conv2d(16, 32, 4, stride=2),
            ]
        )
        self.output = nn.Linear(n_linear_inputs, n_output_channels)

        self.apply(init_chainer_default)
        self.apply(constant_bias_initializer(bias=bias))

    def forward(self, state):
        h = state
        for layer in self.layers:
            h = self.activation(layer(h))
        h_flat = h.view(h.size(0), -1)
        return self.activation(self.output(h_flat))