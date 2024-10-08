#!/bin/bash
MODEL_DIR=${1}
CKPT=${2}
mode=${3}
EVAL_TASK=${4}

# python cliport/eval.py eval_task=pick-and-place-primitive \
#                        agent=cliport \
#                        mode=val \
#                        n_demos=500 \
#                        train_demos=20000 \
#                        checkpoint_type=val_missing \
#                        model_dir=/nfs/gdata/shengqiang/cliport/loho-ravens/ \
#                        exp_folder=exps-rho1-cuda7 \
#                        data_dir=/nfs/gdata/shengqiang/cliport/loho-ravens/data/

echo "---------------- Evaluate on the ${mode} set -----------------"

# python cliport/eval.py model_task=lohoravens-pick-and-place-primitive \
#     eval_task=lohoravens-pick-and-place-primitive \
#     agent=cliport \
#     mode=${mode} \
#     n_demos=200 \
#     train_demos=20000 \
#     checkpoint_type=${CKPT} \
#     type=multi \
#     model_dir=${MODEL_DIR} \
#     exp_folder=exps \
#     data_dir=/mounts/work/shengqiang/projects/2023/LoHoRavens/dataset/ \
# train_config=/nfs/gdata/shengqiang/cliport/loho-ravens/exps-rho1-cuda7/pick-and-place-primitive-cliport-n20000-train/.hydra/config.yaml \
# save_path=/nfs/gdata/shengqiang/cliport/loho-ravens/exps-rho1-cuda7/pick-and-place-primitive-cliport-n20000-train/downstream_eval/${EVAL_TASK}-cliport-n20000-train/ \
# results_path=/nfs/gdata/shengqiang/cliport/loho-ravens/exps-rho1-cuda7/pick-and-place-primitive-cliport-n20000-train/downstream_eval/${EVAL_TASK}-cliport-n20000-train/ \


# multi
python cliport/eval.py model_task=lohoravens-cliport-seen-tasks \
    eval_task=${EVAL_TASK} \
    agent=cliport \
    mode=${mode} \
    n_demos=200 \
    train_demos=1000 \
    checkpoint_type=${CKPT} \
    type=multi \
    model_dir=${MODEL_DIR} \
    exp_folder=exps \
    data_dir=/mounts/work/shengqiang/projects/2023/LoHoRavens/data_v2/unseen \
    record.save_video=False \
    # eval_task=lohoravens-pick-and-place-primitive \

# single
# python cliport/eval.py model_task=pick-and-place-primitive \
#     eval_task=pick-and-place-primitive \
#     agent=cliport \
#     mode=${mode} \
#     n_demos=100 \
#     train_demos=1000 \
#     checkpoint_type=${CKPT} \
#     type=single \
#     model_dir=${MODEL_DIR} \
#     exp_folder=exps \
#     data_dir=/mounts/work/shengqiang/projects/2023/LoHoRavens/dataset/ \
#     # record.save_video=True \
