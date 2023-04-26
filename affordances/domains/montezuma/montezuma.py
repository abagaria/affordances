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
  # 'resources/monte_env_states/room1/enemy/right_of_skull_0.pkl',
  # 'resources/monte_env_states/room1/enemy/left_of_skull_1.pkl'
  # ]
]

old_montezuma_starts = [
  'resources/monte_env_states/room1/ladder/left_top_0.pkl',
  'resources/monte_env_states/room1/ladder/middle_top_0.pkl',
  'resources/monte_env_states/room1/ladder/right_top_0.pkl',
  'resources/monte_env_states/room1/ladder/left_bottom_0.pkl',
  'resources/monte_env_states/room1/ladder/right_bottom_0.pkl',
  'resources/monte_env_states/room1/ladder/middle_bottom_0.pkl',
  'resources/monte_env_states/room1/platforms/under_key_1.pkl',
  'resources/monte_env_states/room1/rope/rope_mid_1.pkl',
  'resources/monte_env_states/room1/rope/rope_bot_1.pkl',
  'resources/monte_env_states/room1/rope/rope_top_1.pkl',
  'resources/monte_env_states/room1/enemy/left_of_skull_0.pkl',
  'resources/monte_env_states/room1/enemy/right_of_skull_0.pkl',
]

new_montezuma_starts = [
  'resources/standard_no_key/x9y235.pkl',
  'resources/standard_no_key/x13y192.pkl',
  'resources/standard_no_key/x13y235.pkl',
  'resources/standard_no_key/x17y192.pkl',
  'resources/standard_no_key/x17y235.pkl',
  'resources/standard_no_key/x21y148.pkl',
  'resources/standard_no_key/x21y152.pkl',
  'resources/standard_no_key/x21y156.pkl',
  'resources/standard_no_key/x21y160.pkl',
  'resources/standard_no_key/x21y164.pkl',
  'resources/standard_no_key/x21y168.pkl',
  'resources/standard_no_key/x21y172.pkl',
  'resources/standard_no_key/x21y176.pkl',
  'resources/standard_no_key/x21y180.pkl',
  'resources/standard_no_key/x21y192.pkl',
  'resources/standard_no_key/x21y235.pkl',
  'resources/standard_no_key/x24y235.pkl',
  'resources/standard_no_key/x25y148.pkl',
  'resources/standard_no_key/x26y235.pkl',
  'resources/standard_no_key/x29y148.pkl',
  'resources/standard_no_key/x30y235.pkl',
  'resources/standard_no_key/x34y235.pkl',
  'resources/standard_no_key/x38y235.pkl',
  'resources/standard_no_key/x42y235.pkl',
  'resources/standard_no_key/x46y235.pkl',
  'resources/standard_no_key/x50y235.pkl',
  'resources/standard_no_key/x69y235.pkl',
  'resources/standard_no_key/x73y235.pkl',
  'resources/standard_no_key/x74y192.pkl',
  'resources/standard_no_key/x77y201.pkl',
  'resources/standard_no_key/x77y205.pkl',
  'resources/standard_no_key/x77y209.pkl',
  'resources/standard_no_key/x77y213.pkl',
  'resources/standard_no_key/x77y217.pkl',
  'resources/standard_no_key/x77y221.pkl',
  'resources/standard_no_key/x77y225.pkl',
  'resources/standard_no_key/x77y229.pkl',
  'resources/standard_no_key/x77y233.pkl',
  'resources/standard_no_key/x77y235.pkl',
  'resources/standard_no_key/x80y192.pkl',
  'resources/standard_no_key/x81y235.pkl',
  'resources/standard_no_key/x85y235.pkl',
  'resources/standard_no_key/x86y192.pkl',
  'resources/standard_no_key/x88y192.pkl',
  'resources/standard_no_key/x104y235.pkl',
  'resources/standard_no_key/x108y235.pkl',
  'resources/standard_no_key/x112y235.pkl',
  'resources/standard_no_key/x116y235.pkl',
  'resources/standard_no_key/x120y235.pkl',
  'resources/standard_no_key/x124y235.pkl',
  'resources/standard_no_key/x125y192.pkl',
  'resources/standard_no_key/x128y235.pkl',
  'resources/standard_no_key/x129y192.pkl',
  'resources/standard_no_key/x130y235.pkl',
  'resources/standard_no_key/x133y148.pkl',
  'resources/standard_no_key/x133y157.pkl',
  'resources/standard_no_key/x133y161.pkl',
  'resources/standard_no_key/x133y165.pkl',
  'resources/standard_no_key/x133y169.pkl',
  'resources/standard_no_key/x133y173.pkl',
  'resources/standard_no_key/x133y177.pkl',
  'resources/standard_no_key/x133y181.pkl',
  'resources/standard_no_key/x133y185.pkl',
  'resources/standard_no_key/x133y192.pkl',
  'resources/standard_no_key/x133y235.pkl',
  'resources/standard_no_key/x137y235.pkl',
  'resources/standard_no_key/x141y235.pkl',
  'resources/standard_no_key/x145y235.pkl',
]

montezuma_subgoals = [
  'resources/monte_env_states/room1/ladder/left_top_0.pkl',
  # 'resources/monte_env_states/room1/ladder/middle_top_0.pkl',
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
