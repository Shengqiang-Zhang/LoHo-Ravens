#!/usr/bin/env bash
TRAIN_LR=${1}
BATCH_SIZE=${2}
GPU_SIZE=${3}
EXP_NOTE=${4}

echo "Train lr: ${TRAIN_LR}"
echo "Train batch size: ${BATCH_SIZE}"
echo "Train gpu size: ${GPU_SIZE}"
echo "Train exp note: ${EXP_NOTE}"

EXP_FOLDER=/mounts/work/shengqiang/projects/2023/LoHoRavens/exps/
ONE_K_DATA_FOLDER=/mounts/work/shengqiang/projects/2023/LoHoRavens/dataset/
TWENTY_K_DATA_FOLDER=/mounts/work/shengqiang/projects/2023/LoHoRavens/pick-and-place-primitive-20k-dataset/
OLD_PICK_PLACE_DATA_FOLDER=/mounts/work/shengqiang/projects/2023/cliport/data/
V2_2K_DATA_FOLDER=/nfs/gdata/shengqiang/cliport/loho-ravens/data_v2/pick-and-place-primitive-2k/
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

python cliport/train.py train.task=pick-and-place-primitive-with-size \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.lr="${TRAIN_LR}" \
                        train.batch_size="${BATCH_SIZE}" \
                        train.gpu="${GPU_SIZE}" \
                        train.n_demos=2000 \
                        train.n_steps=1201000 \
                        train.n_val=100 \
                        train.load_from_last_ckpt=False \
                        dataset.cache=False \
                        train.exp_folder=${EXP_FOLDER} \
                        train.data_dir=${V2_2K_DATA_FOLDER} \
                        dataset.type=single \
                        train.log=True \
                        train.exp_note=${EXP_NOTE} \
                        train.batchnorm=False \
                        tag=${EXP_NOTE} \
                        wandb.logger.project=lohoravens-pick-and-place-primitive \
