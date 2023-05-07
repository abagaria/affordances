onager prelaunch +jobname 5-7-sum \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-7-sum \
    --n_episodes=5000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False" \
    +arg --environment_name DoorCIP SlideCIP LeverCIP \
    +arg --init_learner random binary gvf weighted-binary \
    +arg --seed 1 2 3 4 5 6 7 8 9 10 \
    +arg --optimal_ik False \
    +arg --segment False \
    +arg --sampler sum max \
    +tag --sub_dir