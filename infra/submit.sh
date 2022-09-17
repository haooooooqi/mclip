#!/bin/bash

# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c4`
salt=5p90
queue_file=$HOME/tpus/queue.txt
lock_dir=$HOME/tpus/lock
mkdir -p $lock_dir
set -x
#####################################################################
# after --: options for both training and fine-tuning
# before --: options only for training

echo "~/mae_jax/infra/wrapper.sh maco $salt 128 base imagenet-1k --config.model.loss_type=norm_l2" >> $queue_file
echo "~/mae_jax/infra/wrapper.sh maco $salt 256 large imagenet-1k --config.model.loss_type=norm_l2" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh maco $salt 128 base imagenet-1k" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh maco $salt 256 large imagenet-1k" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh maco $salt 128 base imagenet-1k --config.model.loss_type=cos" >> $queue_file
# for layer in 2 4 12; do
#     echo "~/mae_jax/infra/wrapper.sh maco $salt 128 base imagenet-1k --config.model.decoder.transformer.num_layers=$layer" >> $queue_file
# done

# echo "~/mae_jax/infra/wrapper.sh maco $salt 256 large imagenet-1k --config.model.loss_type=cos" >> $queue_file

######################################################################
set +x
rmdir $lock_dir
