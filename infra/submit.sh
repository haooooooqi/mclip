#!/bin/bash

salt=`head /dev/urandom | tr -dc a-z0-9 | head -c4`
queue_file=$HOME/tpus/queue.txt
lock_dir=$HOME/tpus/lock
mkdir -p $lock_dir
set -x
#####################################################################
# after --: options for both training and fine-tuning
# before --: options only for training

~/mae_jax/infra/wrapper.sh pretrain $salt 128 huge imagenet-1k --mask_sz=32 --image_size=448 -- --patch_sz=16

######################################################################
set +x
rmdir $lock_dir

