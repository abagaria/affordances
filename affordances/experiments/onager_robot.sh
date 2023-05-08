onager prelaunch +jobname 5-8-random \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-8-random \
    --n_episodes=5000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False" \
    +arg --environment_name DoorCIP SlideCIP LeverCIP \
    +arg --init_learner random \
    +arg --seed 1 2 3 4 5 6 7 8 9 10 \
    +arg --optimal_ik False \
    +arg --segment False \
    +arg --sampler max \
    +arg --uncertainty none \
    +tag --sub_dir