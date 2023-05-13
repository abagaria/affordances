---------
# TODO: Door with best 
# 2 * 2 * 3 * 4 * 3
# final runs: 3 * 3 * 2 

onager prelaunch +jobname 5-13-class-count-qpos-slide \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-13-class-count-qpos-slide \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False \
    --eval_accuracy True" \
    +arg --environment_name SlideCIP \
    +arg --init_learner binary weighted-binary \
    +arg --seed 1 2 3 \
    +arg --uncertainty count_qpos \
    +arg --bonus_scale 0.1 0.25 0.5 \
    +arg --gestation 1 5 \
    +tag --sub_dir

onager prelaunch +jobname 5-13-class-count-qpos-lever \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-13-class-count-qpos-lever \
    --n_episodes=5000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False \
    --eval_accuracy True" \
    +arg --environment_name LeverCIP \
    +arg --init_learner binary weighted-binary \
    +arg --seed 1 2 3 \
    +arg --uncertainty count_qpos \
    +arg --bonus_scale 0.1 0.25 0.5 \
    +arg --gestation 1 5 \
    +tag --sub_dir

onager prelaunch +jobname 5-13-class-none-slide \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-13-class-none-slide \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False \
    --eval_accuracy True" \
    +arg --environment_name SlideCIP \
    +arg --init_learner binary weighted-binary \
    +arg --seed 1 2 3 \
    +arg --uncertainty none \
    +arg --bonus_scale 0 \
    +arg --gestation 1 5 \
    +tag --sub_dir

onager prelaunch +jobname 5-13-class-none-lever \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-13-class-none-lever \
    --n_episodes=5000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False \
    --eval_accuracy True" \
    +arg --environment_name LeverCIP \
    +arg --init_learner binary weighted-binary \
    +arg --seed 1 2 3 \
    +arg --uncertainty none \
    +arg --bonus_scale 0 \
    +arg --gestation 1 5 \
    +tag --sub_dir

--------

onager prelaunch +jobname 5-13-gvf \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-13-gvf \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False \
    --eval_accuracy True" \
    +arg --environment_name DoorCIP LeverCIP SlideCIP \
    +arg --init_learner gvf \
    +arg --seed 1 2 3 4 \
    +arg --optimal_ik False \
    +arg --segment False \
    +arg --sampler sum \
    +arg --uncertainty count_qpos \
    +arg --bonus_scale 0.05 0.1 0.25 0.5 \
    +tag --sub_dir

onager prelaunch +jobname 5-13-gvf-none \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-13-gvf \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False \
    --eval_accuracy True" \
    +arg --environment_name DoorCIP LeverCIP SlideCIP \
    +arg --init_learner gvf \
    +arg --seed 1 2 3 4 \
    +arg --optimal_ik False \
    +arg --segment False \
    +arg --sampler sum \
    +arg --uncertainty none \
    +arg --bonus_scale 0 \
    +tag --sub_dir

onager prelaunch +jobname 5-13-class \
    +command "python -m affordances.experiments.robot_td3 \
    --experiment_name=5-13-class \
    --n_episodes=10000 \
    --log_dir=/gpfs/data/gdk/babbatem/ \
    --vis_init_set False \
    --eval_accuracy True" \
    +arg --environment_name DoorCIP LeverCIP SlideCIP \
    +arg --init_learner binary weighted-binary \
    +arg --seed 1 2 3 4 5 \
    +arg --optimal_ik False \
    +arg --segment False \
    +arg --sampler sum \
    +arg --uncertainty none \
    +arg --only_reweigh_negatives True False \
    +arg --gestation 1 5 10 \
    +tag --sub_dir

onager prelaunch +jobname 5-13-gvf \
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