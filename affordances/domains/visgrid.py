import os
import pfrl
import affordances.agent.dqn.wrappers as wrappers

from visgrid.envs.gridworld import GridworldEnv
from visgrid.wrappers.grayscale import GrayscaleWrapper


_path_to_visgrid_saved_mazes = os.path.expanduser("~/git-repos/visgrid/visgrid/envs/saved")


def from_saved_maze(cls, rows: int, cols: int, seed: int, *args, **kw):
    maze_file = f'{_path_to_visgrid_saved_mazes}/mazes_{rows}x{cols}/seed-{seed:03d}/maze-{seed}.txt'
    return cls.from_file(maze_file, *args, **kw)


def environment_builder(use_random_maze, size, seed, test, max_episode_steps=50):
    assert size in (6, 13), size

    if use_random_maze:
        assert size == 6, "random maze is only supported for size=6"
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
        if size == 6:
            dimensions = GridworldEnv.dimensions_6x6_to_64x64
        else:
            dimensions = GridworldEnv.dimensions_13x13_to_84x84
        env = GridworldEnv(
          rows=size,
          cols=size,
          goal_position=(size-1,size-1),
          dimensions=dimensions,
          exploring_starts=not test
        )
    
    env = GrayscaleWrapper(env)
    env = wrappers.Float2UInt8Wrapper(env)
    env = wrappers.UnsqueezeChannelWrapper(env)
    env = wrappers.VisgridWrapper(env)
    env = wrappers.ContinuingTimeLimit(
        env, max_episode_steps=max_episode_steps)
    
    return env
