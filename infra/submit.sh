#!/bin/bash

salt=`head /dev/urandom | tr -dc a-z0-9 | head -c4`
queue_file=$HOME/tpus/queue.txt
lock_dir=$HOME/tpus/lock
mkdir -p $lock_dir
set -x
#####################################################################
# after --: options for both training and fine-tuning
# before --: options only for training

~/mae_jax/infra/wrapper.sh pretrain $salt 128 base imagenet-1k
~/mae_jax/infra/wrapper.sh pretrain $salt 128 base imagenet-1k --config.learning_rate=1.0e-4
~/mae_jax/infra/wrapper.sh pretrain $salt 256 base imagenet-1k --config.num_epochs=1600
~/mae_jax/infra/wrapper.sh pretrain $salt 256 base imagenet-1k --config.learning_rate=1.0e-4 --config.num_epochs=1600

~/mae_jax/infra/wrapper.sh pretrain $salt 128 large imagenet-1k
~/mae_jax/infra/wrapper.sh pretrain $salt 128 large imagenet-1k --config.learning_rate=1.0e-4
~/mae_jax/infra/wrapper.sh pretrain $salt 256 large imagenet-1k --config.num_epochs=1600
~/mae_jax/infra/wrapper.sh pretrain $salt 256 large imagenet-1k --config.learning_rate=1.0e-4 --config.num_epochs=1600

~/mae_jax/infra/wrapper.sh pretrain $salt 256 huge imagenet-1k
~/mae_jax/infra/wrapper.sh pretrain $salt 512 huge imagenet-1k --config.num_epochs=1600
~/mae_jax/infra/wrapper.sh pretrain $salt 256 huge imagenet-1k --config.learning_rate=1.0e-4
~/mae_jax/infra/wrapper.sh pretrain $salt 512 huge imagenet-1k --config.learning_rate=1.0e-4 --config.num_epochs=1600

######################################################################
set +x
rmdir $lock_dir

