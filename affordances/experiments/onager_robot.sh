onager prelaunch +jobname 5-11-slide-class \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-11-slide-class \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False" \
    +arg --environment_name SlideCIP \
    +arg --init_learner binary weighted-binary \
    +arg --seed 1 2 3 4 5 \
    +arg --optimal_ik False \
    +arg --segment False \
    +arg --sampler sum \
    +arg --uncertainty none \
    +arg --only_reweigh_negatives True False \
    +arg --gestation 1 5 10 \
    +tag --sub_dir

onager prelaunch +jobname 5-11-slide-gvf-counts \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-11-slide-gvf-counts \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False" \
    +arg --environment_name SlideCIP \
    +arg --init_learner gvf \
    +arg --seed 1 2 3 4 5 \
    +arg --optimal_ik False \
    +arg --segment False \
    +arg --sampler sum \
    +arg --uncertainty count_qpos count_grasp \
    +arg --bonus_scale 0.05 0.1 0.25 0.5 1.0 \
    +tag --sub_dir

onager prelaunch +jobname 5-11-slide-gvf-none \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-11-slide-gvf-none \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False" \
    +arg --environment_name SlideCIP \
    +arg --init_learner gvf \
    +arg --seed 1 2 3 4 5 \
    +arg --optimal_ik False \
    +arg --segment False \
    +arg --sampler sum \
    +arg --uncertainty none \
    +arg --bonus_scale 0.0 \
    +tag --sub_dir

onager prelaunch +jobname 5-11-slide-rand \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-11-slide-rand \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False" \
    +arg --environment_name SlideCIP \
    +arg --init_learner random \
    +arg --seed 1 2 3 4 5 \
    +arg --optimal_ik False \
    +arg --segment False \
    +tag --sub_dir