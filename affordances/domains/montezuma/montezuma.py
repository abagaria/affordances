import ipdb
from .wrappers import *
from .dopamine_env import create_atari_environment
from .mr_info_wrapper import MontezumaInfoWrapper

from pfrl.wrappers import atari_wrappers


# These are the RAM states we can reset to
environment_rams = [
  [
  'resources/monte_env_states/room1/ladder/left_top_0.pkl',
  'resources/monte_env_states/room1/ladder/left_top_1.pkl',
  'resources/monte_env_states/room1/ladder/middle_top_0.pkl',
  'resources/monte_env_states/room1/ladder/middle_top_1.pkl',
  'resources/monte_env_states/room1/ladder/middle_top_2.pkl',
  'resources/monte_env_states/room1/ladder/middle_top_3.pkl',
  'resources/monte_env_states/room1/ladder/right_top_0.pkl',
  'resources/monte_env_states/room1/ladder/right_top_1.pkl',
  'resources/monte_env_states/room1/ladder/right_top_2.pkl',
  'resources/monte_env_states/room1/ladder/right_top_3.pkl',
  ],[
  'resources/monte_env_states/room1/ladder/left_bottom_0.pkl',
  'resources/monte_env_states/room1/ladder/left_bottom_1.pkl',
  'resources/monte_env_states/room1/ladder/left_bottom_2.pkl',
  'resources/monte_env_states/room1/ladder/middle_bottom_0.pkl',
  'resources/monte_env_states/room1/ladder/middle_bottom_1.pkl',
  'resources/monte_env_states/room1/ladder/middle_bottom_2.pkl',
  'resources/monte_env_states/room1/ladder/right_bottom_0.pkl',
  'resources/monte_env_states/room1/ladder/right_bottom_1.pkl',
  'resources/monte_env_states/room1/ladder/right_bottom_2.pkl',
  ],[
  'resources/monte_env_states/room1/rope/rope_mid_1.pkl',
  'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
  'resources/monte_env_states/room1/rope/rope_top_1.pkl',
  'resources/monte_env_states/room1/rope/rope_top_2.pkl',
  ],[
  'resources/monte_env_states/room1/rope/rope_bot_1.pkl',
  'resources/monte_env_states/room1/rope/rope_bot_2.pkl',
  'resources/monte_env_states/room1/rope/rope_mid_1.pkl',
  'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
  'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
  'resources/monte_env_states/room1/rope/rope_top_1.pkl',
  ],[
  'resources/monte_env_states/room1/platforms/middle_ladder_top_left.pkl',
  'resources/monte_env_states/room1/platforms/right_top_platform_left.pkl',
  'resources/monte_env_states/room1/platforms/middle_right_platform_left.pkl',
  'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
  'resources/monte_env_states/room1/rope/rope_top_1.pkl',
  ],[
  'resources/monte_env_states/room1/platforms/left_top_platform_right.pkl',
  'resources/monte_env_states/room1/platforms/middle_ladder_bottom_right.pkl',
  'resources/monte_env_states/room1/platforms/middle_ladder_top_right.pkl',
  'resources/monte_env_states/room1/rope/rope_mid_2.pkl',
  'resources/monte_env_states/room1/rope/rope_top_1.pkl',
  ],[
  'resources/monte_env_states/room1/platforms/under_key_1.pkl',
  'resources/monte_env_states/room1/platforms/under_key_2.pkl',
  'resources/monte_env_states/room1/platforms/under_key_3.pkl',
  ],
  # [
  # 'resources/monte_env_states/room1/enemy/right_of_skull_0.pkl',
  # 'resources/monte_env_states/room1/enemy/right_of_skull_1.pkl',
  # ],[
  # 'resources/monte_env_states/room1/enemy/left_of_skull_0.pkl',
  # 'resources/monte_env_states/room1/enemy/left_of_skull_1.pkl'
  # ]
]

montezuma_subgoals = [
  'resources/monte_env_states/room1/ladder/left_top_0.pkl',
  'resources/monte_env_states/room1/ladder/middle_top_0.pkl',
  'resources/monte_env_states/room1/ladder/right_top_0.pkl',
  'resources/monte_env_states/room1/ladder/left_bottom_0.pkl',
  'resources/monte_env_states/room1/ladder/right_bottom_0.pkl',
  'resources/monte_env_states/room1/platforms/under_key_1.pkl',
]

# TODO: replace the make_atari with the dopamine env

def environment_builder(max_frames_per_episode):
  env = create_atari_environment()
  return MontezumaInfoWrapper(
    atari_wrappers.FrameStack(
      Reshape(
        RewardClippingWrapper(ContinuingTimeLimit(
          env, max_frames_per_episode
        )), channel_order="chw"
      ),k=4, channel_order="chw"
    )
  )
