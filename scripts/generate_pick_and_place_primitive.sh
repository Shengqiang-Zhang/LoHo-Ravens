# DATA_DIR=/nfs/gdata/shengqiang/cliport/loho-ravens/data_v3/
DATA_DIR=/mounts/work/shengqiang/projects/2023/LoHoRavens/data_v3/pick-and-place-primitives/
DISP=False

echo "Generating dataset... Folder: $DATA_DIR"

# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks

#LANG_TASKS='align-rope assembling-kits-seq-seen-colors assembling-kits-seq-unseen-colors packing-shapes packing-boxes-pairs-seen-colors packing-boxes-pairs-unseen-colors packing-seen-google-objects-seq packing-unseen-google-objects-seq packing-seen-google-objects-group packing-unseen-google-objects-group put-block-in-bowl-seen-colors put-block-in-bowl-unseen-colors stack-block-pyramid-seq-seen-colors stack-block-pyramid-seq-unseen-colors separating-piles-seen-colors separating-piles-unseen-colors towers-of-hanoi-seq-seen-colors towers-of-hanoi-seq-unseen-colors'
#LANG_TASKS='pick-and-place put-all-block-in-a-zone put-block-in-matching-bowl stack-all-block'
#LANG_TASKS='pick-and-place-primitive-with-absolute-position'
LANG_TASKS='pick-and-place-primitive pick-and-place-primitive-with-size pick-and-place-primitive-with-absolute-position'

# LOHORAVENS_TASKS='stack-all-blocks-on-a-zone stack-all-blocks-on-a-zone-with-details stack-blocks-of-same-color  stack-blocks-by-color stack-blocks-of-same-size stack-blocks-by-color-and-size stack-blocks-by-relative-position-and-color stack-blocks-by-absolute-position-and-color-in-size-order move-blocks-between-absolute-positions move-blocks-between-absolute-positions-by-size put-block-into-mismatching-bowl put-blocks-into-matching-bowls-with-details put-hidden-block-into-matching-bowl put-hidden-blocks-in-two-layer-towers-into-matching-bowls'
#stack-smaller-over-bigger-with-same-color
#stack-smaller-over-bigger-with-same-color-in-same-color-zone
#stack-blocks-by-relative-position-and-color-and-size
#stack-blocks-by-absolute-position-and-color-and-size
#
#move-blocks-between-absolute-positions-by-size-and-color
#put-even-blocks-in-same-color-zone put-block-into-matching-bowl
#put-hidden-blocks-in-three-layer-towers-into-matching-bowls
#put-hidden-blocks-in-pyramid-into-matching-bowls
#stack-blocks-with-alternate-color stack-blocks-by-color-in-size-order

for task in $LANG_TASKS
    do
#         python cliport/demos.py n=1000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
#         python cliport/demos.py n=100  task=$task mode=val   data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=200  task=$task mode=test  data_dir=$DATA_DIR disp=$DISP &
    done
echo "Finished Language Tasks."
