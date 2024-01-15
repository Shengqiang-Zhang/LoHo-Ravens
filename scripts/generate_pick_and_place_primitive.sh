DATA_DIR=/nfs/gdata/shengqiang/cliport/loho-ravens/data/
DISP=False

echo "Generating dataset... Folder: $DATA_DIR"

# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks

#LANG_TASKS='align-rope assembling-kits-seq-seen-colors assembling-kits-seq-unseen-colors packing-shapes packing-boxes-pairs-seen-colors packing-boxes-pairs-unseen-colors packing-seen-google-objects-seq packing-unseen-google-objects-seq packing-seen-google-objects-group packing-unseen-google-objects-group put-block-in-bowl-seen-colors put-block-in-bowl-unseen-colors stack-block-pyramid-seq-seen-colors stack-block-pyramid-seq-unseen-colors separating-piles-seen-colors separating-piles-unseen-colors towers-of-hanoi-seq-seen-colors towers-of-hanoi-seq-unseen-colors'
#LANG_TASKS='pick-and-place put-all-block-in-a-zone put-block-in-matching-bowl stack-all-block'
LANG_TASKS='pick-and-place-primitive'

for task in $LANG_TASKS
    do
        python cliport/demos.py n=20000 task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=500  task=$task mode=val   data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=500  task=$task mode=test  data_dir=$DATA_DIR disp=$DISP
    done
echo "Finished Language Tasks."
