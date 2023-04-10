import ipdb
import pfrl
import torch
import numpy as np

from pfrl.utils import evaluating
from pfrl.agents.double_dqn import DoubleDQN
from pfrl.replay_buffers.prioritized import PrioritizedReplayBuffer

from affordances.agent.dqn.utils import batch_experiences


class HierarchicalDoubleDQN(DoubleDQN):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._env = env  # env with options
    
    def get_option_mask(self, state: dict) -> np.ndarray:
        return np.asarray([o.can_run(state) for o in self._env.options])

    def act(self, obs, info):
        return self.batch_act([obs], [info])[0]
    
    def observe(self, obs, reward, done, reset, info):
        return self.batch_observe([obs], [reward], [done], [reset], [info])
    
    def modify_q_values_during_action_selection(self, batch_action_values, masks):
        """Given a single state and mask, modify Q-values of illegal actions."""
        assert len(batch_action_values.q_values) == len(masks)
        assert len(batch_action_values.q_values.shape) == 2, "n_envs, n_actions"
        assert len(masks.shape) == 2, "n_envs, n_actions"
        assert batch_action_values.q_values.shape == masks.shape
        assert masks.shape[1] == self._env.action_space.n, masks.shape
        for i in range(masks.shape[0]):  # n_envs dimension
            min_val = batch_action_values.q_values.min() - 1.
            for j in range(self._env.action_space.n):
                if masks[i, j] == 0:
                    batch_action_values.q_values[i, j] = min_val
        return batch_action_values

    def batch_act(self, batch_obs, batch_info):
        with torch.no_grad(), evaluating(self.model):
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            masks = np.asarray([self.get_option_mask(info) for info in batch_info])
            batch_av = self.modify_q_values_during_action_selection(batch_av, masks)
            greedy_actions = batch_av.q_values.detach().argmax(axis=1).int()
            batch_argmax = greedy_actions.detach().cpu().numpy()  # TODO(ab)
        if self.training:
            batch_action = [
                self.explorer.select_action(
                    self.t,
                    lambda: batch_argmax[i],
                    action_value=batch_av[i : i + 1],
                )
                for i in range(len(batch_obs))
            ]
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax
        return batch_action
    
    def update(self, experiences, errors_out=None):
        """Update the model from experiences

        Args:
            experiences (list): List of lists of dicts.
                For DQN, each dict must contains:
                  - state (object): State
                  - action (object): Action
                  - reward (float): Reward
                  - is_state_terminal (bool): True iff next state is terminal
                  - next_state (object): Next state
                  - weight (float, optional): Weight coefficient. It can be
                    used for importance sampling.
            errors_out (list or None): If set to a list, then TD-errors
                computed from the given experiences are appended to the list.

        Returns:
            None
        """
        has_weight = "weight" in experiences[0][0]
        exp_batch = batch_experiences(
            experiences,
            device=self.device,
            phi=self.phi,
            gamma=self.gamma,
            batch_states=self.batch_states,
        )
        if has_weight:
            exp_batch["weights"] = torch.tensor(
                [elem[0]["weight"] for elem in experiences],
                device=self.device,
                dtype=torch.float32,
            )
            if errors_out is None:
                errors_out = []
        loss = self._compute_loss(exp_batch, errors_out=errors_out)
        if has_weight:
            assert isinstance(self.replay_buffer, PrioritizedReplayBuffer)
            self.replay_buffer.update_errors(errors_out)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            pfrl.utils.clip_l2_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.optim_t += 1
    
    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch["next_state"]
        batch_info = exp_batch["extra_info"]

        with evaluating(self.model):
            next_qout = self.model(batch_next_state)  # next_qout.greedy_actions
            next_qout = self.modify_q_values_during_action_selection(  #TODO(ab): this should be batched
                next_qout,
                masks=np.asarray([self.get_option_mask(info) for info in batch_info])
            )
            next_actions = next_qout.q_values.detach().argmax(dim=1)

        target_next_qout = self.target_model(batch_next_state)
        next_q_max = target_next_qout.evaluate_actions(next_actions)

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_max
    
    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset, batch_info=None) -> None:
        if self.training:
            return self._batch_observe_train(
                batch_obs, batch_reward, batch_done, batch_reset, batch_info
            )
        else:
            return self._batch_observe_eval(
                batch_obs, batch_reward, batch_done, batch_reset
            )
    
    def _batch_observe_train(
        self,
        batch_obs,
        batch_reward,
        batch_done,
        batch_reset,
        batch_info
    ) -> None:

        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1
            # Update the target network
            if self.t % self.target_update_interval == 0:
                self.sync_target_network()
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                    "extra_info": batch_info[i]
                }
                self.replay_buffer.append(env_id=i, **transition)
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)
