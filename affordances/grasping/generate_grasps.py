import os
import time
import argparse 
import sys 
import copy
import random
import pickle 

import numpy as np
import torch
import gym 
import open3d as o3d

import matplotlib.pyplot as plt
plt.plot()
plt.close()

import robosuite as suite
from robosuite.utils import camera_utils
from robosuite.controllers import load_controller_config

from affordances.grasping.GraspSelector import GraspSelector
import affordances.grasping.grasp_pose_generator as gpg
from affordances.utils import utils


RENDER = False

#Element ID info
#DoorCIP: handle is 46
#DrawerCIP: handle is 121
#LeverCIP: handle is 51
#SlideCIP: handle is 51

ELEMENT_IDS = {
    'DoorCIP': [46],
    'DrawerCIP': [121],
    'LeverCIP': [51],
    'SlideCIP': [51],
}

def vertical_flip(img):
    return np.flip(img, axis=0)

def get_task_element_ids(env, task_name, segment):

    # maybe just return handle ID 
    if segment:
        return ELEMENT_IDS[task_name]

    # else get all geoms with name associated with task 
    obj_name = task_name[:3]
    task_elements = []
    for geom in range(len(env.sim.model.geom_type)):
        name = env.sim.model.geom_id2name(geom) 
        if obj_name in name:
            task_elements.append(geom)
            print(name)
    return task_elements


def task2handlePC(env, pointcloud_cameras, task_elements, debug=True):
    obs = env.reset() #Get the raw reset observation from robosuite
    obj_pose = env.get_obj_pose()

    masked_pcd_list = []
    for camera in pointcloud_cameras:
        #vertical flip because OpenGL buffer loading is backwards
        depth_image = vertical_flip(obs[camera+'_depth'])
        rgb_image = vertical_flip(obs[camera+'_image'])
        segmentation_image = vertical_flip(obs[camera+'_segmentation_element'])

        #depth image is normalized by robosuite, this gets real depth map
        depth_image_numpy = camera_utils.get_real_depth_map(env.sim, depth_image)

        if debug:
            f, axarr = plt.subplots(1,3) 
            axarr[0].imshow(rgb_image)
            axarr[1].imshow(depth_image)
            axarr[2].imshow(segmentation_image)
            plt.show()

        all_masked_segmentation = np.where(segmentation_image == task_elements, 1.0, -1.0)
        masked_segmentation = np.max(all_masked_segmentation, axis=-1).reshape(segmentation_image.shape)

        #apply masked segmentation to depth image
        masked_depth_image_numpy = np.multiply(masked_segmentation,depth_image_numpy).astype(np.float32)
        #convert to open3d image
        masked_depth_image = o3d.geometry.Image(masked_depth_image_numpy)

        #Get extrinsics of camera
        extrinsic_cam_parameters= camera_utils.get_camera_extrinsic_matrix(env.sim, camera)

        assert depth_image.shape == segmentation_image.shape

        #All images should have same shape, so we can just use depth image for width and height
        img_width = depth_image.shape[1]
        img_height = depth_image.shape[0]

        intrinisc_cam_parameters_numpy = camera_utils.get_camera_intrinsic_matrix(env.sim, camera, img_width, img_height)
        cx = intrinisc_cam_parameters_numpy[0][2]
        cy = intrinisc_cam_parameters_numpy[1][2]
        fx = intrinisc_cam_parameters_numpy[0][0]
        fy = intrinisc_cam_parameters_numpy[1][1]

        intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic(img_width, #width 
                                                            img_height, #height
                                                            fx,
                                                            fy,
                                                            cx,
                                                            cy)

        masked_pcd = o3d.geometry.PointCloud.create_from_depth_image(masked_depth_image,                                                       
                                              intrinisc_cam_parameters
                                             )

        if len(masked_pcd.points) == 0:
            print("Camera {camera} has no masked points, skipping over".format(camera=camera))
            continue

        #Transform the pcd into world frame based on extrinsics
        masked_pcd.transform(extrinsic_cam_parameters)

        #estimate normals
        masked_pcd.estimate_normals()
        #orientation normals to camera
        masked_pcd.orient_normals_towards_camera_location(extrinsic_cam_parameters[:3,3])

        if debug:
            f, axarr = plt.subplots(2,2) 
            axarr[0,0].imshow(rgb_image)
            axarr[1,0].imshow(depth_image)
            axarr[0,1].imshow(segmentation_image)
            axarr[1,1].imshow(masked_segmentation)

        masked_pcd_list.append(copy.deepcopy(masked_pcd))

    for i in range(len(masked_pcd_list)-1):
        complete_masked_pcd = masked_pcd_list[i] + masked_pcd_list[i+1]
    if (len(masked_pcd_list) == 1):
        complete_masked_pcd = masked_pcd_list[0]

    return complete_masked_pcd, obj_pose

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task',   required=True, type=str, 
                        help='name of task. DrawerCIP, DoorCIP, ...')

    parser.add_argument("--seed", default=0, help="seed",
                        type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument('-n', '--num_samples', type=int, required=True)
    parser.add_argument("--path", default="./affordances/grasping/pointclouds/", type=str, help="location to save point clouds")
    parser.add_argument('--output', type=str, default="./affordances/domains/grasps")
    parser.add_argument('-c','--cameras', nargs='+', default=["birdview","frontview","agentview","sideview"], help='list of cameras to use')
    parser.add_argument('--seg', action="store_true", help="segment pointcloud for handle only?")

    args = parser.parse_args()
    print(args)
    utils.set_random_seed(args.seed)
    pointcloud_cameras = args.cameras

    # create environment instance
    controller_config = load_controller_config(default_controller="OSC_POSE")
    controller_config["impedance_mode"] = "variable_kp"
    controller_config["scale_stiffness"] = True

    # set up env options 
    options = {}
    options["env_name"] = args.task
    options["robots"] = "Panda"
    options["controller_configs"] = controller_config
    options["ee_fixed_to_handle"] = False

    # create and wrap env 
    raw_env = suite.make(
        **options,
        has_renderer=RENDER,
        camera_names=pointcloud_cameras,
        camera_segmentations="element", #element, class, instance
        has_offscreen_renderer=True,
        use_camera_obs=True,
        reward_shaping=True,
        camera_depths=True
    )
    raw_env.deterministic_reset = True 

    task_elements = get_task_element_ids(raw_env, args.task, args.seg)
    test_cloud_with_normals, obj_pose = task2handlePC(raw_env, pointcloud_cameras, task_elements) 

    if args.seg: 
        pcd_fname = f"{args.path}/{args.task}_seg.ply"
        pose_fname = f"{args.path}/{args.task}_pose_seg.npy"
    else:
        pcd_fname = f"{args.path}/{args.task}.ply"
        pose_fname = f"{args.path}/{args.task}_pose.npy"

    o3d.io.write_point_cloud(pcd_fname, test_cloud_with_normals)
    np.save(pose_fname, obj_pose)

    # compute the grasps 
    visualize = False

    world_frame_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    if visualize:
        o3d.visualization.draw_geometries([test_cloud_with_normals, world_frame_axes])
    assert(np.asarray(test_cloud_with_normals.normals).shape == np.asarray(test_cloud_with_normals.points).shape)

    gs = GraspSelector(obj_pose, test_cloud_with_normals)
    sampled_poses = gs.getRankedGraspPoses()
    random.shuffle(sampled_poses)
    desired_sampled_poses = sampled_poses[:args.num_samples]
    desired_sampled_poses = [gpg.translateFrameNegativeZ(p, gs.dist_from_point_to_ee_link) for p in desired_sampled_poses]

    if args.seg: 
        grasp_fname = f"{args.output}/{args.task}_seg.pkl"
    else:
        grasp_fname = f"{args.output}/{args.task}.pkl"
        
    pickle.dump(desired_sampled_poses, open(grasp_fname,"wb"))
    # gs.visualizeGraspPoses(desired_sampled_poses)


