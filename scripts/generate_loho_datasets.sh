#!/bin/bash

DATA_DIR=/mounts/work/shengqiang/projects/2023/LoHoRavens/pick-and-place-primitive-dataset/
DISP=False

echo "Generating dataset... Folder: $DATA_DIR"

# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks

# pick-and-place primitive
LANG_TASKS='pick-and-place-primitive
    pick-and-place-primitive-with-size
    pick-and-place-primitive-with-absolute-position'

for task in $LANG_TASKS; do
#    python cliport/demos.py n=20000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
    python cliport/demos.py n=200 task=$task mode=val data_dir=$DATA_DIR disp=$DISP &
    python cliport/demos.py n=200 task=$task mode=test data_dir=$DATA_DIR disp=$DISP &
done

# put-block-in-bowl
#LANG_TASKS='put-block-in-matching-bowl
#    put-block-in-mismatching-bowl
#    stack-smaller-over-bigger-with-same-color
#    stack-block-in-absolute-area'
#
#LANG_TASKS='put-even-block-in-same-color-zone'

# LANG_TASKS='put-block-in-matching-bowl
#     stack-smaller-over-bigger-with-same-color
#     stack-block-in-absolute-area'
# 
# for task in $LANG_TASKS; do
#     python cliport/demos.py n=1000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
#     python cliport/demos.py n=200 task=$task mode=val data_dir=$DATA_DIR disp=$DISP &
#     python cliport/demos.py n=200 task=$task mode=test data_dir=$DATA_DIR disp=$DISP &
# done

#LANG_TASKS='put-block-in-mismatching-bowl
#    stack-block-of-same-size
#    stack-smaller-over-bigger-with-same-color-in-same-color-zone
#    move-block-in-x-area-to-y-area
#    stack-block-of-same-color'
#
#
#for task in $LANG_TASKS; do
##    python cliport/demos.py n=1000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
#    python cliport/demos.py n=200 task=$task mode=val data_dir=$DATA_DIR disp=$DISP &
#    python cliport/demos.py n=200 task=$task mode=test data_dir=$DATA_DIR disp=$DISP &
#done
