#!/bin/bash

EVAL_TASK="pick-and-place"

python cliport/eval.py eval_task=${EVAL_TASK} \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=20000 \
                       checkpoint_type=last \
                       model_dir=/nfs/gdata/shengqiang/cliport/loho-ravens/ \
                       exp_folder=exps-rho1-cuda7 \
                       data_dir=/mounts/work/shengqiang/projects/2023/cliport/loho-ravens-seen-tasks/ \
                       model_path=/nfs/gdata/shengqiang/cliport/loho-ravens/exps-rho1-cuda7/pick-and-place-primitive-cliport-n20000-train/checkpoints/ \
                       train_config=/nfs/gdata/shengqiang/cliport/loho-ravens/exps-rho1-cuda7/pick-and-place-primitive-cliport-n20000-train/.hydra/config.yaml \
                       save_path=/nfs/gdata/shengqiang/cliport/loho-ravens/exps-rho1-cuda7/pick-and-place-primitive-cliport-n20000-train/downstream_eval/${EVAL_TASK}-cliport-n20000-train/ \
                       results_path=/nfs/gdata/shengqiang/cliport/loho-ravens/exps-rho1-cuda7/pick-and-place-primitive-cliport-n20000-train/downstream_eval/${EVAL_TASK}-cliport-n20000-train/ \
