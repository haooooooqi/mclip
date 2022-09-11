#!/bin/bash

# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c4`
salt=4p90
queue_file=$HOME/tpus/queue.txt
lock_dir=$HOME/tpus/lock
mkdir -p $lock_dir
set -x
#####################################################################
# after --: options for both training and fine-tuning
# before --: options only for training

for nc in 2 4 5; do
    echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.num_crops=$nc" >> $queue_file
    echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.num_crops=$nc --config.model.encoder.num_decoder_layer=0" >> $queue_file
done

echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.num_crops=2 --config.model.encoder.num_decoder_layer=0" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.batch_size=4096 --config.learning_rate 1e-4 --config.model.encoder.num_decoder_layer=0" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 huge imagenet-1k --config.model.encoder.num_decoder_layer=0" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.lr_schedule=linear" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.min_abs_lr=8e-5" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.opt.ema_momentum=0." >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.encoder.num_decoder_layer=0" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 512 large imagenet-1k --config.model.encoder.num_decoder_layer=0" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.batch_size=4096 --config.learning_rate 1e-4" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.batch_size=4096 --config.learning_rate 1e-4" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.encoder.num_decoder_layer=1" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.encoder.num_decoder_layer=1" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.aug.area_min=0.08" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.aug.area_min=0.08" >> $queue_file

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
