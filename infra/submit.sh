#!/bin/bash

# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c4`
salt=7p90
queue_file=$HOME/tpus/queue.txt
lock_dir=$HOME/tpus/lock
mkdir -p $lock_dir
set -x
#####################################################################
# after --: options for both training and fine-tuning
# before --: options only for training

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.encoder.num_decoder_layer=8" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.model.loss_type=cos --config.model.encoder.decoder_type=class --config.model.encoder.num_decoder_layer=4" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k" >> $queue_file

# for lw in .1 .3 3.; do
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.intra_weight=$lw" >> $queue_file
# done

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.encoder.decoder_type=class" >> $queue_file

# for nl in 4 8; do
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.model.loss_type=cos --config.model.encoder.num_decoder_layer=$nl" >> $queue_file
# done

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.model.loss_type=cos" >> $queue_file

# for nq in 2 4; do
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.model.loss_type=cos --config.model.encoder.num_queries=$nq" >> $queue_file
# done

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.num_epochs=100" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.num_epochs=200" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.num_epochs=400" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.num_epochs=800" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.loss_type=cos --config.model.encoder.num_decoder_layer=0" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.loss_type=cos --config.model.encoder.num_decoder_layer=0 --config.model.num_crops=1" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.loss_type=cos" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.loss_type=cos --config.model.num_crops=1" >> $queue_file

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.model.loss_type=cos" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.model.loss_type=cos --config.model.num_crops=1" >> $queue_file

# for am in .08 .2 .5; do
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.aug.area_min=$am" >> $queue_file
# done

# for b2 in .95 .99 .999; do
#     for mmt in .99 .996; do
#         echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 base imagenet-1k --config.opt.b2=$b2 --config.opt.ema_momentum=$mmt" >> $queue_file
#     done
# done

# for temp in .1 .3; do
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.temp=$temp" >> $queue_file
# done

# for nc in 2 4 5; do
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.num_crops=$nc" >> $queue_file
#     echo "~/mae_jax/infra/wrapper.sh mclr $salt 128 base imagenet-1k --config.model.num_crops=$nc --config.model.encoder.num_decoder_layer=0" >> $queue_file
# done

# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.num_crops=2 --config.model.encoder.num_decoder_layer=0" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.num_crops=2" >> $queue_file
# echo "~/mae_jax/infra/wrapper.sh mclr $salt 256 large imagenet-1k --config.model.num_crops=2 --config.opt.ema_schedule=cos" >> $queue_file

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
