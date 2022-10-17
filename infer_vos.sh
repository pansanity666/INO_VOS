#!/bin/bash

{
DS=$1 # [davis|ytvos]

# we provide default test parameters here.
# you can try different parameters as you need.
case $DS in
ytvos)
  echo "Test dataset: YouTube-VOS 2018 (val)"
  FILELIST=./meta/val_ytvos2018_test.txt
  SCALE_SIZE=-1 # Do not change this 
  ARCH='vit_small' 
  TOPK=5 # N_k
  N_LAST_FRAMES=20 # N_c
  R=50 # N_r 
  N_LAYER=6 # Test with last N layer output faetures 
  ;;
davis)
  echo "Test dataset: DAVIS-2017 (val)"
  FILELIST=./meta/val_davis2017_test.txt
  SCALE_SIZE=480 # Do not change this 
  ARCH='vit_small'
  TOPK=5 # N_k
  N_LAST_FRAMES=10  # N_c
  R=40 # N_r 
  N_LAYER=6 # Test with last N layer output faetures 
  ;;
*)
  echo "Dataset '$DS' not recognised. Should be one of [ytvos|davis]."
  exit 1
  ;;
esac  

CKP_PATH="$2" # ./charades_best.pth
MSK_CACHE_PATH="./cached/masks"  

SAVE_NAME="[${TOPK},${N_LAST_FRAMES},${R}]_layer${N_LAYER}"

# inference result path
ckp_name=$(basename $CKP_PATH)
SAVE_DIR="./results/$ckp_name/$DS/$SAVE_NAME"

# mkdir before run 
if [ ! -d $SAVE_DIR ]; then
  echo "Creating directory: $SAVE_DIR"
  mkdir -p $SAVE_DIR
else
  echo "Saving to: $SAVE_DIR"
fi

# 
CMD="python infer_vos.py   --resume $CKP_PATH \
                           --infer-list $FILELIST \
                           --mask-output-dir $SAVE_DIR \
                           --mask_cache_path $MSK_CACHE_PATH \
                           --scale_size $SCALE_SIZE \
                           --img_size 64 \
                           --arch ${ARCH} \
                           --topk ${TOPK} \
                           --n_last_frames ${N_LAST_FRAMES} \
                           --size_mask_neighborhood ${R} \
                           --test_last_n_layer ${N_LAYER} \
                           $EXTRA"
echo $CMD > ${SAVE_DIR}.cmd 
$CMD 
# nohup $CMD > $LOG_FILE 2>&1 &

# ===== do davis evaluation after generating =====
# case $DS in 
# davis)
# python /mnt4/panxiao.pan/remote157_mnt/workspace/dataset/davis2017-evaluation/evaluation_method.py \
#  --task semi-supervised \
#  --davis_path ./data/DAVIS/ \
#  --results_path "${SAVE_DIR}/vos"
# ;;
# esac

exit

}