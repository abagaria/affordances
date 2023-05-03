onager prelaunch +jobname robot_5_3_gvf \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=robot_5_3_gvf \
    --n_episodes=5000 \
    --log_dir=/gpfs/data/gdk/babbatem/" \
    +arg --environment_name DoorCIP LeverCIP SlideCIP \
    +arg --init_learner gvf weighted-binary \
    +arg --seed 1 2 3 4 5 6 7 8 9 10 \
    +arg --optimal_ik True False \
    +arg --segment False \
    +tag --sub_dir