import os 
import copy 
import pickle 
from PIL import Image

import gym
import numpy as np 
from tqdm import tqdm

from gym import spaces 

import robosuite as suite
from robosuite.controllers import load_controller_config

def make_robosuite_env(task, 
    deterministic=True, 
    render=False, 
    use_qpos_cache=False, 
    pregrasp=True, 
    optimal_ik=True, 
    learning=True, 
    segment=True):
    
    # create environment instance
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp"
    controller_config["scale_stiffness"] = True
    controller_config["safety_bool"] = False
    controller_config["action_scale_param"] = 0.5

    options = {}
    options["env_name"] = task
    options["robots"] = "Panda"
    options["controller_configs"] = controller_config
    options["ee_fixed_to_handle"] = True
    options["manip_strategy"] = "old"
    options["p_constant"] = 1
    options["m_constant"] = 1
    options["ttt_constant"] = 1
    options["horizon"] = 250
    options["reward_shaping"] = False
    options["hard_reset"] = True if not render else False

    # create and wrap env 
    raw_env = suite.make(
        **options,
        has_renderer=render,
        has_offscreen_renderer=not render,
        use_camera_obs=False,
    )
    
    if not deterministic:
        assert use_qpos_cache == False

    raw_env.deterministic_reset = deterministic
    env = RobotEnvWrapper(raw_env, 
                           pregrasp_policy=pregrasp, 
                           learning=learning,
                           terminate_when_lost_contact=True,
                           num_steps_lost_contact=500,
                           optimal_ik=optimal_ik,
                           control_gripper=True,
                           use_qpos_cache=use_qpos_cache,
                           segment=segment)
    return env 

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
                 control_gripper=False,
                 use_qpos_cache=False,
                 segment=True):

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
        self.learning = learning 
        self.optimal_ik = optimal_ik
        self.terminate_when_lost_contact = terminate_when_lost_contact
        self.control_gripper = control_gripper
        self.qpos_cache = {} 
        self.use_qpos_cache = use_qpos_cache
        self.segment = segment

    def get_states_from_grasps(self, n=1):
        states = []
        qpos = []
        for grasp in tqdm(self.grasp_list):
            for _ in range(n):
                obs, ik_soln = self.reset_to(grasp)
                states.append(obs)
                qpos.append(ik_soln)
        return np.array(states), np.array(qpos)

    def load_grasps(self):
        task = self.env.__class__.__name__
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        
        if self.segment:
            heuristic_grasps_path = cur_dir + "/grasps/"+task+"_filtered_seg.pkl"
        else:
            heuristic_grasps_path = cur_dir + "/grasps/"+task+"_filtered.pkl"

        heuristic_grasps = pickle.load(open(heuristic_grasps_path,"rb"))
        grasp_list, self.grasp_wp_scores, self.grasp_qpos_list = list(zip(*heuristic_grasps))
        grasp_list = np.array(grasp_list)
        self.grasp_list = grasp_list
        return grasp_list

    def reset_to_idx(self, idx):
        return self.reset_to(self.grasp_list[idx])

    def reset_to(self, sampled_pose):
        obs = super().reset()
        self.pre_grasp_complete = False
        self.contact_hist = [True]*self.num_steps_lost_contact
        
        ik_soln = None 
        if sampled_pose is not None:

            while True:
                self.grasp_success = self.reset_to_grasp(
                    sampled_pose, wide=True, 
                    optimal_ik=self.optimal_ik,
                    frame=self.get_obj_pose(),
                    verbose=self.env.has_renderer
                )
                if self.grasp_success or not self.learning:
                    break

            ik_soln = copy.deepcopy(self.sim.data.qpos[:7])
            if self.pregrasp_policy:
                obs = self.execute_pregrasp()

        return obs, ik_soln

    def reset_to_joint_state(self, qpos):
        obs = super().reset()
        self.pre_grasp_complete = False
        self.contact_hist = [True]*self.num_steps_lost_contact
        self.grasp_success = self.reset_to_qpos(qpos, wide=True)

        assert self.grasp_success
        if self.pregrasp_policy:
            obs = self.execute_pregrasp()
        return obs


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
        obs = np.concatenate((obs["robot0_proprio-state"],obs["object-state"]))
        return obs.astype(np.float32)
    
    def step(self, action):
        if not self.control_gripper:
            action[-1] = 1
        
        obs, reward, done, info = super().step(action)
        if self.env.has_renderer:
            self.render()

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

        info["needs_reset"] = done
        return obs, reward, done, info

    def render_states(self, states, vis=False, fpath=None, ep_num=0):
        for i, state in enumerate(states):
            self.reset_to_qpos(state, wide=True)
            img = self.sim.render(
                camera_name='frontview',
                width=256,
                height=256
            )
            img = img[::-1]

            if vis:
                plt.imshow(img)
                plt.show()

            if fpath is not None:
                im = Image.fromarray(img)
                fname = f"ep{str(ep_num).zfill(4)}_qpos{str(i).zfill(3)}.png"
                im.save(os.path.join(fpath, fname))


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    plt.figure()
    plt.close()
    from affordances.utils import utils
    utils.set_random_seed(0)

    # env = make_robosuite_env("DoorCIP", render=True, segment=True)
    # env = make_robosuite_env("LeverCIP", render=True, segment=True)
    # env = make_robosuite_env("SlideCIP", render=True, segment=True)
    # env = make_robosuite_env("DrawerCIP", render=True, segment=True)

    # env = make_robosuite_env("DoorCIP", render=True, segment=False) # good
    # env = make_robosuite_env("LeverCIP", render=True, segment=False) # good
    # env = make_robosuite_env("SlideCIP", render=True, segment=False) # good
    # env = make_robosuite_env("DrawerCIP", render=True, segment=False) # n=3

    # testing rendering of init set 
    env = make_robosuite_env("DoorCIP", render=False, segment=False) 
    qpos_per_grasp = 1
    grasps = env.load_grasps()
    grasp_state_vectors, grasp_qpos = env.get_states_from_grasps(n=qpos_per_grasp)
    env.render_states(grasp_qpos, fpath="./results/dev", vis=True)

    for i in range(len(grasps)):
        print(i % len(grasps))
        env.reset_to(grasps[i % len(grasps)])
        for j in range(10):
            obs, rew, done, info = env.step(np.zeros(13))
            env.render()