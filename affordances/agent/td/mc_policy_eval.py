"""Monte-carlo policy evaluation."""

import gym
from collections import defaultdict, deque


class EveryVisitTabularMC:
  """Every-visit tabular Monte-Carlo policy evaluation."""

  def __init__(
      self,
      positions_to_track: set,
      hash_key: str = 'player_pos',
      discount_factor: float = 0.99,
      maintain_samples: bool = True
  ):
    self.counts_table = defaultdict(lambda: defaultdict(float))
    self.cumulative_value_table = defaultdict(lambda: defaultdict(float))
    
    self.hash_key = hash_key
    self.gamma = discount_factor
    self.maintain_samples = maintain_samples
    self.positions_to_track = positions_to_track
    
    # position tuple -> list of 10 obs/image samples
    self.pos2obs = defaultdict(lambda: deque([], maxlen=10))

  def get_value(self, info, goal_info):
    key1 = self._get_key(info)
    key2 = self._get_key(goal_info)
    cum_reward = self.cumulative_value_table[key1][key2]
    count = self.counts_table[key1][key2]
    if count > 0:
      return cum_reward / count
    return 1.

  def update(self, trajectory, goal_info):
    """Update the value-table using the goal-conditioned trajectory."""
    key2 = self._get_key(goal_info)
    for i, transition in enumerate(trajectory):
      key1 = self._get_key(transition[-1])
      if key1 in self.positions_to_track:
        g_t = self._discounted_return(trajectory, i)
        self.counts_table[key1][key2] += 1
        self.cumulative_value_table[key1][key2] += g_t

        if self.maintain_samples:
          obs = transition[3]
          assert obs.shape == (1, 84, 84), obs.shape
          self.pos2obs[key1].append(obs)
      
  def _discounted_return(self, trajectory, idx):
    """Sum of discounted goal-conditioned rewards in the rest of the traj."""
    rewards = [trans[2] for trans in trajectory[idx:]]
    return sum([(self.gamma**i)*r for i, r in enumerate(rewards)])

  @staticmethod
  def _get_key(info):
    return info if isinstance(info, tuple) else info['player_pos']


if __name__ == '__main__':
  desc = [
    "SFFF",
    "FFFF",
    "FFFF",
    "FFFG",
  ]
  goal_pos = (15,)
  env = gym.make('FrozenLake-v1', desc=desc, is_slippery=False)
  policy_evaluator = EveryVisitTabularMC(
    positions_to_track=[(x,) for x in range(16)],
  )
  print(policy_evaluator.positions_to_track)
  for episode in range(1000):
    done = False
    trajectory = []
    obs = env.reset()
    episode_reward = 0
    while not done:
      action = env.action_space.sample()
      next_obs, reward, done, info = env.step(action)
      info['player_pos'] = (next_obs,)
      trajectory.append((obs, action, reward, next_obs, done, info))
      obs = next_obs
      episode_reward += reward
    print(f'Episode {episode} Reward {episode_reward}')
    policy_evaluator.update(trajectory, goal_pos)
