#!/usr/bin/env bash
TRAIN_LR=${1}
echo "Train lr: ${TRAIN_LR}"
#EXP_FOLDER=/nfs/gdata/shengqiang/cliport/loho-ravens/exps/
#DATA_FOLDER=/nfs/gdata/shengqiang/cliport/loho-ravens/data/
#
#python cliport/train.py train.task=pick-and-place-primitive \
#                        train.agent=cliport \
#                        train.attn_stream_fusion_type=add \
#                        train.trans_stream_fusion_type=conv \
#                        train.lang_fusion_type=mult \
#                        train.n_demos=20000 \
#                        train.n_steps=200000 \
#                        train.exp_folder=${EXP_FOLDER} \
#                        train.data_dir=${DATA_FOLDER} \
#                        dataset.cache=False

# python cliport/train.py train.task=pick-and-place-primitive \
#                         train.agent=cliport \
#                         train.attn_stream_fusion_type=add \
#                         train.trans_stream_fusion_type=conv \
#                         train.lang_fusion_type=mult \
#                         train.n_demos=20000 \
#                         train.n_steps=200000 \
#                         train.exp_folder=/nfs/gdata/shengqiang/cliport/loho-ravens/exps/ \
#                         train.data_dir=/nfs/gdata/shengqiang/cliport/loho-ravens/data/ \
#                         dataset.cache=False \
#                         train.log=True \

EXP_FOLDER=/mounts/work/shengqiang/projects/2023/LoHoRavens/exps/
DATA_FOLDER=/mounts/work/shengqiang/projects/2023/LoHoRavens/dataset/

#python cliport/train.py train.task=pick-and-place-primitive \
#                        train.agent=cliport \
#                        train.attn_stream_fusion_type=add \
#                        train.trans_stream_fusion_type=conv \
#                        train.lang_fusion_type=mult \
#                        train.n_demos=20000 \
#                        train.n_steps=200000 \
#                        train.exp_folder=${EXP_FOLDER} \
#                        train.data_dir=${DATA_FOLDER} \
#                        dataset.cache=False

python cliport/train.py train.task=lohoravens-cliport-seen-tasks \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.lr="${TRAIN_LR}" \
                        train.n_demos=1000 \
                        train.n_steps=351000 \
                        train.n_val=200 \
                        dataset.cache=False \
                        train.exp_folder=${EXP_FOLDER} \
                        train.data_dir=${DATA_FOLDER} \
                        dataset.type=multi \
                        train.log=True
