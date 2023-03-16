import gym
import numpy as np 

from gym import spaces 

class Spec(object):
    def __init__(self,id):
        super().__init__()
        self.id = id

class RobotEnvWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env, 
                 pregrasp_policy=True, 
                 save_torques=False,
                 learning=True,
                 terminate_when_lost_contact=False,
                 num_steps_lost_contact=500,
                 optimal_ik=False,
                 control_gripper=False,):

        # setup action space 
        low_limits = env.action_spec[0]
        high_limits = env.action_spec[1]
        a_space_shape = env.action_spec[0].shape
        env.action_space = spaces.Box(low=low_limits, high=high_limits, shape=a_space_shape, dtype=np.float32)

        # compute obs shape
        obs = self.observation(env.reset())
        obs_shape = obs.shape
        env.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=obs_shape, dtype=np.float32)
        env.reward_range = (-float("inf"), float("inf"))
        env.metadata = {}
        env.spec = Spec("RobotEnvWrapper")
        super().__init__(env)

        self.env = env
        self.pregrasp_policy = pregrasp_policy
        self.num_broken = 0
        self.num_steps_lost_contact = num_steps_lost_contact
        self._max_episode_steps = 250
        self.grasp_strategy=grasp_strategy
        self.learning = learning 
        self.use_cached_qpos = use_cached_qpos 
        self.optimal_ik = optimal_ik
        self.terminate_when_lost_contact = terminate_when_lost_contact
        self.control_gripper = control_gripper
        self.optimal_ik = optimal_ik

    def get_states_from_grasps(self):
        pass

    def reset_to(self, sampled_pose):

        self.pre_grasp_complete = False
        self.contact_hist = [True]*self.num_steps_lost_contact

        if sampled_pose is not None:
            keep_resetting = True
            while keep_resetting:
                o = super().reset()
                self.grasp_success = self.reset_to_grasp(
                                        sampled_pose, wide=True, 
                                        optimal_ik=self.optimal_ik,
                                        frame=self.get_obj_pose(),
                                        verbose=self.env.has_renderer
                                     )
                if self.grasp_success:   
                    keep_resetting = False

                if not self.learning:
                    keep_resetting = False
        else:
            o = super().reset()
            self.sim.forward()

        if self.pregrasp_policy:
            o = self.execute_pregrasp()
        return(o)

    def execute_pregrasp(self):
        # close gripper for a few frames
        a = np.zeros(self.env.action_spec[0].shape)
        a[-1] = 1
        for _ in range(10):
            o, r, d, i = self.step(a)

        self.pre_grasp_complete = True
        return o 

    def set_render(self, render_state):
        self.env.has_renderer = render_state
    def render(self, mode=None):
        self.env.render()
    def observation(self, obs):
        return(np.concatenate((obs["robot0_proprio-state"],obs["object-state"])))
    
    def step(self, action):
        if not self.control_gripper:
            action[-1] = 1
        
        obs, reward, done, info = super().step(action)
        if self.env.has_renderer:
            self.render()

        # check arm qpos
        # TODO: tolerance is hardcoded 0.1 rad
        self.torque_history.append(self.env.robots[0].torques)
        lost_contact = False 
        if self.pre_grasp_complete and self.terminate_when_lost_contact:

            self.contact_hist.append(self.check_gripper_contact(self.env.__class__.__name__))
            last_five_contacts = self.contact_hist[-self.num_steps_lost_contact:]
            lost_contact = not np.any(last_five_contacts)
            if lost_contact:
                done = True
                success = False 
            info["lost_contact"] = lost_contact

        unsafe = False
        if self.env.robots[0].check_q_limits():
            done = True
            unsafe = True 
            success = False 
            info["unsafe_qpos"] = True
        info["unsafe_qpos"] = unsafe

        success = self.env._check_success()
        if success:
            done = True
        info["is_success"] = success 

        if done and not self.is_eval_env:
            self.init_learner.add_rollout(success)
        
        return obs, reward, done, info
            

    def seed(self, seed=None):
        """ set numpy seed etc. directly instead. """
        pass
