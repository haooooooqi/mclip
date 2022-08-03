#!/bin/bash

salt=`head /dev/urandom | tr -dc a-z0-9 | head -c4`
queue_file=$HOME/tpus/queue.txt
lock_dir=$HOME/tpus/lock
mkdir -p $lock_dir
set -x
#####################################################################
# after --: options for both training and fine-tuning
# before --: options only for training

# echo "~/mask/wrapper.sh train_in1k $salt 512 huge --mask_sz=32 --image_size=448 -- --patch_sz=16" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 512 large --train_epochs=1600 -- --mask_dim=768 --mask_dim_mlp=3072 --mask_heads=12 --mask_pred_blocks=12" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 huge --train_epochs=1600 -- --mask_dim=768 --mask_dim_mlp=3072 --mask_heads=12 --mask_pred_blocks=12" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 huge --train_epochs=1600 --mask_sz=16 -- --patch_sz=16" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 huge --mask_sz=16 -- --patch_sz=16" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 512 giant_mae --train_epochs=1600" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 512 large --mask_target=org --mask_sz=32 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 large --mask_target=org --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_target=org --mask_sz=32 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_target=org --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --mask_target=org --mask_sz=32 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --mask_target=org --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 512 base --mask_target=org --mask_sz=32 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 base --mask_target=org --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 512 huge_mae --train_epochs=1600" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 huge_mae --train_epochs=1600 --mask_sz=28 --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --train_epochs=16000" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --train_epochs=4000 --mask_sz=4 -- --image_size=56 --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --train_epochs=4000 --mask_sz=8 -- --image_size=112 --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --train_epochs=4000 -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --train_epochs=4000 --mask_sz=4 -- --image_size=56 --crop_interp=area --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --train_epochs=4000 --mask_sz=8 -- --image_size=112 --crop_interp=area --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 base --train_epochs=4000 -- --crop_interp=area --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 256 base" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=2 --mask_target=layer_norm_mask -- --image_size=28 --patch_sz=1" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=4 --mask_target=layer_norm_mask -- --image_size=56 --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=8 --mask_target=layer_norm_mask -- --image_size=112 --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_target=layer_norm_mask -- --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=2 --mask_target=layer_norm_mask -- --image_size=28 --crop_interp=area --patch_sz=1" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=4 --mask_target=layer_norm_mask -- --image_size=56 --crop_interp=area --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=8 --mask_target=layer_norm_mask -- --image_size=112 --crop_interp=area --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_target=layer_norm_mask -- --crop_interp=area --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=4 -- --image_size=56 --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=8 -- --image_size=112 --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base -- --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=4 -- --image_size=56 --crop_interp=area --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=8 -- --image_size=112 --crop_interp=area --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base -- --crop_interp=area --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_target=org --mask_sz=2 -- --image_size=28 --crop_interp=area --patch_sz=1" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_target=org --mask_sz=4 -- --image_size=56 --crop_interp=area --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_target=org --mask_sz=8 -- --image_size=112 --crop_interp=area --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_target=org -- --crop_interp=area --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_sz=2 -- --image_size=28 --crop_interp=area --patch_sz=1" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_sz=4 -- --image_size=56 --crop_interp=area --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_sz=8 -- --image_size=112 --crop_interp=area --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large -- --crop_interp=area --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_sz=2 -- --image_size=28 --patch_sz=1" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_sz=4 -- --image_size=56 --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_sz=8 -- --image_size=112 --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large -- --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 256 base --mask_sz=4 -- --image_size=56 --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 256 base --mask_sz=8 -- --image_size=112 --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 256 base -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 256 base --mask_sz=4 -- --image_size=56 --crop_interp=area --patch_sz=2" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 256 base --mask_sz=8 -- --image_size=112 --crop_interp=area --patch_sz=4" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 256 base -- --crop_interp=area --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 256 huge_mae" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 huge_mae -- --patch_sz=7" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 huge_mae --mask_sz=28 --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 256 huge_mae" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge_mae -- --patch_sz=7" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge_mae --mask_sz=28 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 huge_mae --train_epochs=4000" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge_mae --train_epochs=4000 -- --patch_sz=7" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge_mae --train_epochs=4000 --mask_sz=28 --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge_mae -- --patch_sz=7" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge_mae --train_epochs=1600 -- --patch_sz=7" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge_mae --mask_sz=28 -- --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 huge_mae" >> $queue_file

# for data in places_challenge places_std in1k coco ; do
#     echo "~/mask/wrapper.sh train_${data} $salt 256 base --mask_sz=8 -- --patch_sz=8" >> $queue_file
#     echo "~/mask/wrapper.sh train_${data} $salt 256 large --mask_sz=8 -- --patch_sz=8" >> $queue_file
# done

# echo "~/mask/wrapper.sh train_places_std $salt 256 large --mask_sz=32 -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_places_std $salt 256 large -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco $salt 256 large --mask_sz=32 -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco $salt 256 base --mask_sz=32 -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 small --mask_sz=32 -- --patch_sz=8" >> $queue_file

# for data in coco_more places_challenge places_std coco; do
#     echo "~/mask/wrapper.sh train_${data} $salt 128 base" >> $queue_file
#     echo "~/mask/wrapper.sh train_${data} $salt 128 base --mask_sz=32" >> $queue_file
#     echo "~/mask/wrapper.sh train_${data} $salt 256 base -- --patch_sz=8" >> $queue_file
#     echo "~/mask/wrapper.sh train_${data} $salt 256 base --mask_sz=32 -- --patch_sz=8" >> $queue_file

#     echo "~/mask/wrapper.sh train_${data} $salt 256 large" >> $queue_file
#     echo "~/mask/wrapper.sh train_${data} $salt 256 large --mask_sz=32" >> $queue_file
#     echo "~/mask/wrapper.sh train_${data} $salt 256 large -- --patch_sz=8" >> $queue_file
#     echo "~/mask/wrapper.sh train_${data} $salt 256 large --mask_sz=32 -- --patch_sz=8" >> $queue_file
# done

# echo "~/mask/wrapper.sh train_coco_more $salt 128 base" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 large" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 256 base --mask_sz=32 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 large --mask_sz=32 --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 128 small --mask_sz=32 -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 large --mask_sz=32 -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 base --mask_sz=32 -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 huge --mask_sz=28 -- --patch_sz=7" >> $queue_file

# echo "~/mask/wrapper.sh train_coco $salt 128 small --mask_sz=32 -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 128 small --mask_sz=32 -- --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 128 small -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 base -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 large -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge -- --patch_sz=7" >> $queue_file

# echo "~/mask/wrapper.sh train_coco $salt 128 small -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco $salt 256 base -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco $salt 512 large -- --patch_sz=8" >> $queue_file
# echo "~/mask/wrapper.sh train_coco $salt 512 large -- --patch_sz=8" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 512 huge --mask_sz=28 --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_places_challenge $salt 128 base --mask_sz=32 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_places_std $salt 256 base --mask_sz=32 --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_places_std $salt 128 base --train_epochs=1200" >> $queue_file
# echo "~/mask/wrapper.sh train_places_challenge $salt 256 base --train_epochs=240" >> $queue_file

# echo "~/mask/wrapper.sh train_places_std $salt 256 large --train_epochs=1200" >> $queue_file
# echo "~/mask/wrapper.sh train_places_challenge $salt 256 large --train_epochs=240" >> $queue_file

# ~/mask/wrapper.sh train_coco_more $salt 128 base

# echo "~/mask/wrapper.sh train_places_challenge $salt 128 base" >> $queue_file
# echo "~/mask/wrapper.sh train_places_std $salt 128 base" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 128 base" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 256 large" >> $queue_file
# echo "~/mask/wrapper.sh train_places_challenge $salt 512 large --mask_sz=32 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_places_std $salt 256 large --mask_sz=32 --image_size=448" >> $queue_file

# ~/mask/wrapper.sh train_places_challenge $salt 256 large
# echo "~/mask/wrapper.sh train_places_std $salt 256 large" >> $queue_file

# echo "~/mask/wrapper.sh train_in1k $salt 128 smallI2 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 256 baseI2 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_in1k $salt 512 largeI2 --image_size=448" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 128 smallCC2 --train_epochs=3200" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 128 smallCC2" >> $queue_file

# echo "~/mask/wrapper.sh train_coco_more $salt 256 smallCC2 --mask_sz=32 --train_epochs=3200 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 smallCC2 --mask_sz=32 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 128 baseCC2 --mask_sz=32 --train_epochs=3200 --image_size=448" >> $queue_file
# echo "~/mask/wrapper.sh train_coco_more $salt 256 largeCC2 --mask_sz=32 --train_epochs=3200 --image_size=448" >> $queue_file

# for ep in 2400 3200 4000 4800; do
#     echo "~/mask/wrapper.sh train_coco_more $salt 128 baseCC2 --train_epochs=$ep" >> $queue_file
# done

# for ep in 2400 4000 4800; do
#     echo "~/mask/wrapper.sh train_coco_more $salt 256 largeCC2 --train_epochs=$ep" >> $queue_file
# done

# for ratio in .8 .85; do
#     echo "~/mask/wrapper.sh train_in1k $salt 512 largeI2 --mask_ratio=$ratio --mask_sz=32 -- --image_size=448" >> $queue_file
# done

######################################################################
set +x
rmdir $lock_dir

