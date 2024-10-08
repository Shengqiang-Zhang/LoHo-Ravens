#!/usr/bin/env bash

CKPT=${1}

GPU1=MIG-f3649993-0669-5e8a-8220-10ccda377d74
GPU2=MIG-f3649993-0669-5e8a-8220-10ccda377d74
GPU3=MIG-e6c9b2f7-e93e-5f29-9ab1-864c18ba8261

LOHORAVENS_TASKS_1='stack-all-blocks-on-a-zone stack-all-blocks-on-a-zone-with-details stack-blocks-of-same-color  stack-blocks-by-color stack-blocks-of-same-size'


#  stack-blocks-by-color-and-size stack-blocks-by-relative-position-and-color stack-blocks-by-absolute-position-and-color-in-size-order move-blocks-between-absolute-positions move-blocks-between-absolute-positions-by-size put-block-into-mismatching-bowl put-blocks-into-matching-bowls-with-details put-hidden-block-into-matching-bowl put-hidden-blocks-in-two-layer-towers-into-matching-bowls'


LOHORAVENS_TASKS_2='stack-blocks-by-color-and-size stack-blocks-by-relative-position-and-color stack-blocks-by-absolute-position-and-color-in-size-order move-blocks-between-absolute-positions'

LOHORAVENS_TASKS_3='move-blocks-between-absolute-positions-by-size put-block-into-mismatching-bowl put-blocks-into-matching-bowls-with-details put-hidden-block-into-matching-bowl put-hidden-blocks-in-two-layer-towers-into-matching-bowls'


UNSEEN_TASKS_1='stack-blocks-with-alternate-color stack-blocks-by-color-in-size-order stack-smaller-over-bigger-with-same-color'

UNSEEN_TASKS_2='stack-smaller-over-bigger-with-same-color-in-same-color-zone stack-blocks-by-relative-position-and-color-and-size stack-blocks-by-absolute-position-and-color-and-size'
UNSEEN_TASKS_3=' move-blocks-between-absolute-positions-by-size-and-color put-hidden-blocks-in-three-layer-towers-into-matching-bowls put-hidden-blocks-in-pyramid-into-matching-bowls'

for t in ${UNSEEN_TASKS_3}
do
    echo "evaluating the task: ${t}"
    CUDA_VISIBLE_DEVICES=${GPU3} \
        bash scripts/eval_imitation_learning.sh \
        /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-cliport-seen-tasks-cliport-n1000-train-batchify/gpu2-bs25-lr0.0001-bs25x2_multiseen_demo1k_step100k_lr1e-4_nobn \
        ${CKPT} \
        test \
        ${t} | tee -a results/eval_imitation_learning_last_unseen_june6_third_part.txt
done



# echo "Train pick-and-place-primitive"
# CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/gpu4-bs25-lr0.0001-bs25x4_multi_demo20k_step12000k_lr1e-4_nobn/ ${CKPT} train pick-and-place-primitive;
# echo "Test pick-and-place-primitive-with-size"
# CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/${MODEL} ${CKPT} test pick-and-place-primitive-with-size
# # echo "Train pick-and-place-primitive-with-size"
# # CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/gpu4-bs25-lr0.0001-bs25x4_multi_demo20k_step12000k_lr1e-4_nobn/ ${CKPT} train pick-and-place-primitive-with-size;
# echo "Test pick-and-place-primitive-with-absolute-position"
# CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/${MODEL} ${CKPT} test pick-and-place-primitive-with-absolute-position
# # echo "Train pick-and-place-primitive-with-absolute-position"
# # CUDA_VISIBLE_DEVICES=${GPU} bash scripts/eval_pick_and_place_primitive_multi.sh /mounts/work/shengqiang/projects/2023/LoHoRavens/exps/lohoravens-pick-and-place-primitive-cliport-n20000-train-batchify/gpu4-bs25-lr0.0001-bs25x4_multi_demo20k_step12000k_lr1e-4_nobn/ ${CKPT} train pick-and-place-primitive-with-absolute-position;