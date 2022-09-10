#!/bin/bash

# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c4`
salt=3p90
queue_file=$HOME/tpus/queue.txt
lock_dir=$HOME/tpus/lock
mkdir -p $lock_dir
set -x
#####################################################################
# after --: options for both training and fine-tuning
# before --: options only for training

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.warmup_epochs=10" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.warmup_epochs=10" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.warmup_epochs=10" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.batch_size=4096" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 large imagenet-1k --config.batch_size=4096" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.batch_size=4096" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.opt.ema_schedule=cos" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.opt.ema_schedule=cos" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.opt.ema_schedule=cos" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.loss_type=info-nce" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.loss_type=info-nce" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.model.loss_type=info-nce" >> $queue_file

######################################################################
set +x
rmdir $lock_dir
