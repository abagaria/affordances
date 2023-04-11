import ipdb
import gym
import argparse
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pfrl
from pfrl import experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN

import affordances.agent.dqn.runloops as runloops
import affordances.agent.dqn.wrappers as wrappers
from affordances.agent.dqn.hddqn import HierarchicalDoubleDQN
from affordances.agent.dqn.utils import RandomizeAction


class SingleSharedBias(nn.Module):
    """Single shared bias used in the Double DQN paper.

    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.

    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

    def __call__(self, x):
        return x + self.bias.expand_as(x)


def parse_arch(arch, n_actions):
    if arch == "nature":
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "doubledqn":
        # raise NotImplementedError("Single shared bias not implemented yet")
        return nn.Sequential(
            pnn.SmallAtariCNN(),
            init_chainer_default(nn.Linear(256, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    elif arch == "nips":
        return nn.Sequential(
            pnn.SmallAtariCNN(),
            init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "dueling":
        return DuelingDQN(n_actions)
    elif arch == "visgrid84":
        return nn.Sequential(
            pnn.SmallAtariCNN(n_input_channels=1),
            init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == 'visgrid64':
        return nn.Sequential(
            pnn.SmallAtariCNN(n_input_channels=1, n_linear_inputs=1152),
            init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )
    else:
        raise RuntimeError("Not supported architecture: {}".format(arch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--final-exploration-frames", type=int, default=10**6)
    parser.add_argument("--final-epsilon", type=float, default=0.01)
    parser.add_argument("--eval-epsilon", type=float, default=0.001)
    parser.add_argument("--noisy-net-sigma", type=float, default=None)
    parser.add_argument(
        "--arch",
        type=str,
        default="doubledqn",
        choices=["nature", "nips", "dueling", "doubledqn", "visgrid64", "visgrid84"],
    )
    parser.add_argument("--steps", type=int, default=5 * 10**7)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--replay-start-size", type=int, default=5 * 10**4)
    parser.add_argument("--target-update-interval", type=int, default=3 * 10**4)
    parser.add_argument("--eval-interval", type=int, default=10**5)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--eval-n-runs", type=int, default=10)
    parser.add_argument("--no-clip-delta", dest="clip_delta", action="store_false")
    parser.set_defaults(clip_delta=True)
    parser.add_argument(
        "--agent", type=str, default="DoubleDQN", choices=["DQN", "DoubleDQN", "PAL"]
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--prioritized",
        action="store_true",
        default=False,
        help="Use prioritized experience replay.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--n-step-return", type=int, default=1)
    parser.add_argument("--use_random_maze", action="store_true", default=False)
    parser.add_argument("--use_global_init_fn", action="store_true", default=False)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_montezuma(seed, test):
        from gym_montezuma.envs.montezuma_env import make_monte_env_as_atari_deepmind
        env = make_monte_env_as_atari_deepmind(
            max_episode_steps=4000,
            episode_life=False,
            clip_rewards=not test,
            frame_skip=4,
            frame_stack=4,  # TODO: dont stack if using pfrl's VevEnvs
            frame_warp=(84,84),
            render_option_execution=False
        )
        env = wrappers.MontezumaWrapper(env)
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = RandomizeAction(env, args.eval_epsilon)
        env.seed(seed)
        return env

    def make_gridworld(seed, test):
        from affordances.domains.visgrid import environment_builder
        return environment_builder(use_random_maze=args.use_random_maze,
                                   seed=seed, test=test) 

    train_env = make_gridworld(args.seed, test=False)
    eval_env = make_gridworld(args.seed, test=True)

    n_actions = train_env.action_space.n
    q_func = parse_arch(args.arch, n_actions)

    if args.noisy_net_sigma is not None:
        pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()
    else:
        explorer = explorers.LinearDecayEpsilonGreedy(
            1.0,
            args.final_epsilon,
            args.final_exploration_frames,
            random_action_func=lambda: train_env.action_space.sample() \
                if args.use_global_init_fn else train_env.sample_random_action(),
        )

    # Use the same hyper parameters as the Nature paper's
    opt = optim.RMSprop(
        q_func.parameters(),
        lr=args.lr,
        alpha=0.95,
        momentum=0.0,
        eps=1e-2,
        centered=True,
    )

    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffers.PrioritizedReplayBuffer(
            10**6,
            alpha=0.6,
            beta0=0.4,
            betasteps=betasteps,
            num_steps=args.n_step_return,
        )
    else:
        rbuf = replay_buffers.ReplayBuffer(10**6, num_steps=args.n_step_return)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = HierarchicalDoubleDQN(
        q_function=q_func,
        optimizer=opt,
        replay_buffer=rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        clip_delta=args.clip_delta,
        update_interval=args.update_interval,
        batch_accumulator="sum",
        phi=phi,
        env=train_env,
        use_dummy_init_fn=args.use_global_init_fn,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        runloops.train_agent_with_evaluation(
            agent=agent,
            env=train_env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=False,
            eval_env=eval_env,
        )
