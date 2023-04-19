import gym
import numpy as np
from gym import spaces
from collections import deque
from pfrl.wrappers.atari_wrappers import LazyFrames


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order="hwc"):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.stack_axis = {"hwc": 2, "chw": 0}[channel_order]
        orig_obs_space = env.observation_space
        low = np.repeat(orig_obs_space.low, k, axis=self.stack_axis)
        high = np.repeat(orig_obs_space.high, k, axis=self.stack_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=orig_obs_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action, clf):
        ob, reward, done, info = self.env.step(action, clf)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames), stack_axis=self.stack_axis)


class Reshape(gym.ObservationWrapper):
    def __init__(self, env, channel_order="hwc"):
        super().__init__(env)
        self.width = 84
        self.height = 84
        shape = {
            "hwc": (self.height, self.width, 1),
            "chw": (1, self.height, self.width),
        }
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=np.uint8
        )

    def observation(self, frame):
        return frame.reshape(self.observation_space.low.shape)

    

class ContinuingTimeLimit(gym.Wrapper):
    """TimeLimit wrapper for continuing environments.

    This is similar gym.wrappers.TimeLimit, which sets a time limit for
    each episode, except that done=False is returned and that
    info['needs_reset'] is set to True when past the limit.

    Code that calls env.step is responsible for checking the info dict, the
    fourth returned value, and resetting the env if it has the 'needs_reset'
    key and its value is True.

    Args:
        env (gym.Env): Env to wrap.
        max_episode_steps (int): Maximum number of timesteps during an episode,
            after which the env needs a reset.
    """

    def __init__(self, env, max_episode_steps):
        super(ContinuingTimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._max_episode_steps <= self._elapsed_steps:
            info["needs_reset"] = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.clip(reward, -1, 1.)
