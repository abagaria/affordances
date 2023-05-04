onager prelaunch +jobname robot_5_4_vis_door \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=robot_5_4_vis \
    --n_episodes=5000 \
    --log_dir=/gpfs/data/gdk/babbatem/" \
    +arg --environment_name DoorCIP \
    +arg --init_learner binary gvf weighted-binary \
    +arg --seed 1 2 3 4 5 6 7 8 9 10 \
    +arg --optimal_ik False \
    +arg --segment False \
    +tag --sub_dir