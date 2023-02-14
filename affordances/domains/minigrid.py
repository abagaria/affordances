import numpy as np
from PIL import Image
import gymnasium as gym
from gymnasium.core import Wrapper, ObservationWrapper
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper, ReseedWrapper


class MinigridInfoWrapper(Wrapper):
  """Include extra information in the info dict for debugging/visualizations."""

  def reset(self):
    obs, info = self.env.reset()
    info = self._modify_info_dict(info)
    return obs, info

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    info = self._modify_info_dict(info, terminated, truncated)
    done = terminated or truncated
    return obs, reward, done, info

  def _modify_info_dict(self, info, terminated=False, truncated=False):
    info['player_pos'] = self.env.agent_pos
    info['player_x'] = self.env.agent_pos[0]
    info['player_y'] = self.env.agent_pos[1]
    info['truncated'] = truncated
    info['terminated'] = terminated
    info['needs_reset'] = truncated  # pfrl needs this flag
    return info


class ResizeObsWrapper(ObservationWrapper):
  """Resize the observation image to be (84, 84) and compatible with Atari."""
  def observation(self, observation):
    img = Image.fromarray(observation)
    return np.asarray(img.resize((84, 84), Image.BILINEAR))


class TransposeObsWrapper(ObservationWrapper):
  def observation(self, observation):
    assert len(observation.shape) == 3, observation.shape
    assert observation.shape[-1] == 3, observation.shape
    return observation.transpose((2, 0, 1))


class SparseRewardWrapper(Wrapper):
  """Return a reward of 1 when you reach the goal and 0 otherwise."""
  def step(self, action):
    # minigrid discounts the reward with a step count - undo that here
    obs, reward, terminated, truncated, info = self.env.step(action)
    return obs, float(reward > 0), terminated, truncated, info


class GrayscaleWrapper(ObservationWrapper):
  def observation(self, observation):
    observation = observation.mean(axis=0)[np.newaxis, :, :]
    return observation.astype(np.uint8)


def determine_goal_pos(env):
  """Convinence hacky function to determine the goal location."""
  from minigrid.core.world_object import Goal
  for i in range(env.grid.width):
    for j in range(env.grid.height):
      tile = env.grid.get(i, j)
      if isinstance(tile, Goal):
          return i, j


def environment_builder(
  level_name='MiniGrid-Empty-8x8-v0',
  reward_fn='sparse',
  grayscale=True
):
  env = gym.make(level_name)
  env = ReseedWrapper(env)  # To fix the goal distribution
  env = RGBImgObsWrapper(env) # Get pixel observations
  env = ImgObsWrapper(env) # Get rid of the 'mission' field
  if reward_fn == 'sparse':
    env = SparseRewardWrapper(env)
  env = ResizeObsWrapper(env)
  env = TransposeObsWrapper(env)
  if grayscale:
    env = GrayscaleWrapper(env)
  env = MinigridInfoWrapper(env)
  return env
