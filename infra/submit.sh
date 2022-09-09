#!/bin/bash

# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c4`
salt=sep9
queue_file=$HOME/tpus/queue.txt
lock_dir=$HOME/tpus/lock
mkdir -p $lock_dir
set -x
#####################################################################
# after --: options for both training and fine-tuning
# before --: options only for training

echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 default imagenet-1k" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 base imagenet-1k --config.seed=0" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 large imagenet-1k --config.seed=0" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 512 huge imagenet-1k --config.seed=0" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 base imagenet-1k --config.seed=2" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 base imagenet-1k --config.seed=2 --config.num_epochs=1600" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 large imagenet-1k --config.seed=2" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 large imagenet-1k --config.seed=2 --config.num_epochs=1600" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 huge imagenet-1k --config.seed=2" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 512 huge imagenet-1k --config.seed=2 --config.num_epochs=1600" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 base imagenet-1k --config.seed=2 --config.lr_schedule=linear" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 base imagenet-1k --config.seed=2 --config.lr_schedule=linear --config.num_epochs=1600" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 large imagenet-1k --config.seed=2 --config.lr_schedule=linear" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 large imagenet-1k --config.seed=2 --config.lr_schedule=linear --config.num_epochs=1600" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 huge imagenet-1k --config.seed=2 --config.lr_schedule=linear" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 512 huge imagenet-1k --config.seed=2 --config.lr_schedule=linear --config.num_epochs=1600" >> $queue_file

# for tau in .1 .2; do
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.learning_rate=5.0e-5 --config.model.clr.tau=$tau --config.model.clr.stop_key=True" >> $queue_file
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.learning_rate=5.0e-5 --config.model.clr.tau=$tau --config.model.clr.stop_key=True" >> $queue_file
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.learning_rate=5.0e-5 --config.model.clr.tau=$tau --config.model.clr.stop_key=True" >> $queue_file
# done

# for ep in 100 300; do
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.num_epochs=$ep" >> $queue_file
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.num_epochs=$ep" >> $queue_file
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.num_epochs=$ep" >> $queue_file
# done

# for ratio in .75; do
#     for tau in .1 .2; do
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.mask_ratio=$ratio --config.model.clr.tau=$tau --config.model.clr.stop_key=True" >> $queue_file
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.mask_ratio=$ratio --config.model.clr.tau=$tau --config.model.clr.stop_key=True" >> $queue_file
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.model.mask_ratio=$ratio --config.model.clr.tau=$tau --config.model.clr.stop_key=True" >> $queue_file
#     done
# done

# for ratio in .7 .9; do
#     for tau in .1 .2; do
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.mask_ratio=$ratio --config.model.clr.tau=$tau" >> $queue_file
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.mask_ratio=$ratio --config.model.clr.tau=$tau" >> $queue_file
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.model.mask_ratio=$ratio --config.model.clr.tau=$tau" >> $queue_file
#     done
# done

# for lr in 1.5e-4 1e-4; do
#     for wd in .3 .5; do
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.learning_rate=$lr --config.opt.weight_decay=$wd" >> $queue_file
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.learning_rate=$lr --config.opt.weight_decay=$wd" >> $queue_file
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.learning_rate=$lr --config.opt.weight_decay=$wd" >> $queue_file
#     done
# done

# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 base imagenet-1k" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 base imagenet-1k --config.learning_rate=1.0e-4" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 base imagenet-1k --config.num_epochs=1600" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 base imagenet-1k --config.learning_rate=1.0e-4 --config.num_epochs=1600" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 large imagenet-1k" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 128 large imagenet-1k --config.learning_rate=1.0e-4" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 large imagenet-1k --config.num_epochs=1600" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 large imagenet-1k --config.learning_rate=1.0e-4 --config.num_epochs=1600" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 huge imagenet-1k" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 512 huge imagenet-1k --config.num_epochs=1600" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 256 huge imagenet-1k --config.learning_rate=1.0e-4" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mae $salt 512 huge imagenet-1k --config.learning_rate=1.0e-4 --config.num_epochs=1600" >> $queue_file

######################################################################
set +x
rmdir $lock_dir
