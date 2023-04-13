import gym
import random
import numpy as np


class MontezumaWrapper(gym.Wrapper):
    def get_current_state(self):
        try:  # TODO: come on
            return self.env.env.env.env.env._state
        except:
            return self.env.env.env.env.env.env._state
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state_dict = self.get_current_state()
        if info.get('needs_reset', False):
            state_dict.update({'needs_reset': True})
        del info
        state_dict.pop('env')
        return obs, reward, done, state_dict
    
    def reset(self):
        obs0 = super().reset()
        s0 = self.get_current_state()
        return obs0, s0
    

class VisgridWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timestep = 0

    def get_current_state(self) -> dict:
        state_array = self.get_state()
        return dict(player_pos=tuple(state_array[:2]))

    def can_execute_action(self, state: dict, action: int) -> bool:
        position = state['player_pos']
        direction = self.unwrapped._action_offsets[action]
        return not self.grid.has_wall(position, direction)
    
    def sample_random_action(self) -> int:
        state = self.get_current_state()
        actions = [a for a in self.options if self.can_execute_action(state, a)]
        return random.choice(actions)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state_dict = self.get_current_state()
        info['bonus'] = 0
        info['needs_reset'] = truncated
        info['terminated'] = terminated
        info['player_pos'] = state_dict['player_pos']
        info['player_x'] = info['player_pos'][0]
        info['player_y'] = info['player_pos'][1]
        info['timestep'] = self._timestep
        self._timestep += 1
        return obs, reward, terminated or truncated, info
    
    def reset(self, pos=None):
        obs, info = self.env.reset(pos=pos)
        info['bonus'] = 0
        state_dict = self.get_current_state()
        info['player_pos'] = state_dict['player_pos']
        info['player_x'] = info['player_pos'][0]
        info['player_y'] = info['player_pos'][1]
        info['needs_reset'] = False
        info['terminated'] = False
        info['timestep'] = self._timestep
        return obs, info

    @property
    def options(self):
        return list(range(self.action_space.n))
    
class UnsqueezeChannelWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return observation[np.newaxis, ...]
    

class Float2UInt8Wrapper(gym.ObservationWrapper):
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255,
                              shape=(1,64,64), dtype=np.uint8)
    
    def observation(self, obs):
        assert obs.dtype == np.float32, obs.dtype
        assert obs.max() < 1.1, obs.max()
        assert obs.min() >= 0, obs.min()
        return (obs * 255).clip(0, 255).astype(np.uint8)


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

    def reset(self, pos=None):
        self._elapsed_steps = 0
        return self.env.reset(pos=pos)
