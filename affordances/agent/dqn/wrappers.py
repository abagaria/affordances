import gym
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
    def get_current_state(self) -> dict:
        state_array = self.get_state()
        return dict(player_pos=tuple(state_array[:2]))

    def can_execute_action(self, state: dict, action: int) -> bool:
        position = state['player_pos']
        direction = self.unwrapped._action_offsets[action]
        return not self.grid.has_wall(position, direction)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state_dict = self.get_current_state()
        info['needs_reset'] = truncated
        info['player_pos'] = state_dict['player_pos']
        return obs, reward, terminated or truncated, info
    
    def reset(self):
        obs, info = self.env.reset()
        info['player_pos'] = self.get_current_state()['player_pos']
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
