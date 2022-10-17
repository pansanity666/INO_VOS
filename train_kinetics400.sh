# modify this as you need.
ckpt_output_path='/mnt4/panxiao.pan/remote83_mnt3/workspace/result/sscl_dataloader_updated_final' # path for saving ckpt and training logs.

# 
dataset_cache_path='./cached/charades' # path for caching dataset meta.
csv_path='./meta/charades_train_list.csv' # path for the csv file of trianing data. 

# ===== Charades from scratch =====
# 8 cards
PORT=${PORT:-29512}
CARD=(0,1,2,3,4,5,6,7)
NGPU=8
BS=4

# 4 cards
# PORT=${PORT:-29511}
# CARD=(0,1,2,3)
# NGPU=4
# BS=4

# Arch
arch=vit_small
patch_size=8

# lr schdule
lr=3e-4
warmup_epochs=2
total_epoch=25
final_lr=1e-6 
start_epoch=0
saveckp_freq=1 # save ckpt interval (EPOCH)

# losses
# 1. out-generative learning loss (We merge g2g and l2g as cfdino.)
weight_cfdino=1.0
# 2. in-generative learning loss: in_mim loss
weight_mim=1.0
# 3. in-generative learning loss: in_aff loss
weight_affine=1.0

# CMD 
CUDA_VISIBLE_DEVICES=$CARD  python -m torch.distributed.launch --nproc_per_node=$NGPU --master_port=$PORT main_VOS.py   \
--dataset 'k400_vrw' --csv_path ${csv_path}   --cache_path  ${dataset_cache_path} --multi_scale_layer 1 \
--arch ${arch} --patch_size ${patch_size}  \
--batch_size_per_gpu $BS --lr ${lr} --warmup_epochs ${warmup_epochs} --epochs ${total_epoch} --min_lr ${final_lr} --start_epoch ${start_epoch} \
--weight_cfdino ${weight_cfdino} --weight_affine ${weight_affine} --weight_mim ${weight_mim} \
--saveckp_freq ${saveckp_freq} --output_dir ${ckpt_output_path} \
--distributed \
