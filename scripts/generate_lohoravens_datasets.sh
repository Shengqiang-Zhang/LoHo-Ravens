#!/bin/bash

DATA_DIR=/mounts/work/shengqiang/projects/2023/LoHoRavens/data_v3/easy/
DISP=False

echo "Generating dataset... Folder: $DATA_DIR"

# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks

# pick-and-place primitive
LANG_TASKS='pick-and-place-primitive
    pick-and-place-primitive-with-size
    pick-and-place-primitive-with-absolute-position'


MOVE_BLOCKS_TASKS='move-blocks-between-absolute-positions
                   move-blocks-between-absolute-positions-by-color
                   move-blocks-between-absolute-positions-by-size
                   move-blocks-between-absolute-positions-by-size-and-color'

STACK_BLOCKS_TASKS='put-even-blocks-in-same-color-zone
stack-blocks-by-absolute-position-and-color-in-size-order
put-hidden-blocks-in-pyramid-into-matching-bowls
put-hidden-blocks-in-three-layer-towers-into-matching-bowls
'

for task in $STACK_BLOCKS_TASKS; do
#    python cliport/demos.py n=20000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
    python cliport/demos.py n=100 task=$task mode=val data_dir=$DATA_DIR disp=$DISP &
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
