import os
import pfrl
import affordances.agent.dqn.wrappers as wrappers

from visgrid.envs.gridworld import GridworldEnv
from visgrid.wrappers.grayscale import GrayscaleWrapper


_path_to_visgrid_saved_mazes = os.path.expanduser("~/git-repos/visgrid/visgrid/envs/saved")


def from_saved_maze(cls, rows: int, cols: int, seed: int, *args, **kw):
    maze_file = f'{_path_to_visgrid_saved_mazes}/mazes_{rows}x{cols}/seed-{seed:03d}/maze-{seed}.txt'
    return cls.from_file(maze_file, *args, **kw)


def environment_builder(use_random_maze, seed, test, max_episode_steps=50):
    if use_random_maze:
        env = from_saved_maze(
            GridworldEnv,
            rows=6,
            cols=6,
            seed=seed,
            goal_position=(5,5),
            agent_position=(0,0),
            dimensions=GridworldEnv.dimensions_6x6_to_64x64,
        )
    else:  # open grid world
        env = GridworldEnv(
          rows=13,
          cols=13,
          goal_position=(12,12),
          dimensions=GridworldEnv.dimensions_13x13_to_84x84,
          exploring_starts=not test
        )
    
    env = GrayscaleWrapper(env)
    env = wrappers.Float2UInt8Wrapper(env)
    env = wrappers.UnsqueezeChannelWrapper(env)
    env = wrappers.VisgridWrapper(env)
    env = wrappers.ContinuingTimeLimit(
        env, max_episode_steps=max_episode_steps)
    
    return env
