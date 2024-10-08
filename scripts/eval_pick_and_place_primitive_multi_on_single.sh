#!/usr/bin/env bash
GPU=${1}
MODEL=${2}
CKPT=${3}

echo "Test pick-and-place-primitive"
CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/${MODEL} ${CKPT} test pick-and-place-primitive
# echo "Train pick-and-place-primitive"
# CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/gpu4-bs25-lr0.0001-bs25x4_multi_demo20k_step12000k_lr1e-4_nobn/ ${CKPT} train pick-and-place-primitive;
echo "Test pick-and-place-primitive-with-size"
CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/${MODEL} ${CKPT} test pick-and-place-primitive-with-size
# echo "Train pick-and-place-primitive-with-size"
# CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/gpu4-bs25-lr0.0001-bs25x4_multi_demo20k_step12000k_lr1e-4_nobn/ ${CKPT} train pick-and-place-primitive-with-size;
echo "Test pick-and-place-primitive-with-absolute-position"
CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/${MODEL} ${CKPT} test pick-and-place-primitive-with-absolute-position
# echo "Train pick-and-place-primitive-with-absolute-position"
# CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/gpu4-bs25-lr0.0001-bs25x4_multi_demo20k_step12000k_lr1e-4_nobn/ ${CKPT} train pick-and-place-primitive-with-absolute-position;
