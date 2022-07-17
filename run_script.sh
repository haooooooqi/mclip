rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_vit_large.py \
    --config.batch_size=8 \
    --config.log_every_steps=10 \
    --config.num_epochs=1000 \
    --config.profile_memory=True \
    --config.donate=True \
    --config.aug.randerase.on=True \
    --config.aug.randerase.prob=0.25 \
    --config.model.rescale_head_init=0.001 \
    --config.aug.mix.mixup=True \
    --config.aug.mix.cutmix=True \
    --config.aug.autoaug=autoaug \
    --config.eval_only=False \
    --config.model.classifier=token \
    --config.learning_rate_decay=0.75 \
    --config.partitioning.num_partitions=1 \
    --config.opt_type=adamw \
    --config.model.sincos=True \
    --config.model.transformer.droppath_rate=0.0 \
    --config.model.predictor.transformer.droppath_rate=0.2 \
    --config.model.predictor.transformer.num_layers=0 \
    --config.model.adapter.on_use=True \
    --config.partitioning.partition_states=True \
    --config.model.stopgrad_blocks=0 \
    --config.partitioning.activation_partitioning_dims=2 \
    --config.partitioning.parameter_partitioning_dims=2 \
    --config.torchload.data_dir='/datasets03/inaturalist/2019' \
    --config.aug.image_size=224 \
    --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220527_064059_maet5x_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_b4096_lr1e-4_mk0.75_s100_p1_normpix_exwd_adamw32' \
    --config.pretrain_fmt=t5x \
    --config.model.canonical_grid=14 \
    --config.model.transformer.renew_layers=0 \
    --config.model.transformer.inter_layers=4 \

    # --config.torchload.data_dir='/datasets/imagenet-22k' \
    # --config.model.num_classes=21841 \

    # --config.torchload.data_dir='/datasets03/inaturalist/2017' \
    # --config.model.num_classes=5089 \
    # --config.torchload.data_dir='/datasets03/inaturalist/2018' \
    # --config.model.num_classes=8142 \
    # --config.torchload.data_dir='/datasets03/inaturalist/2019' \
    # --config.model.num_classes=1010 \


    # --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220527_064059_maet5x_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_b4096_lr1e-4_mk0.75_s100_p1_normpix_exwd_adamw32_finetune/20220622_010421_kmh-tpuvm-v3-256-3_cfg_vit_large_50ep_fttl_IN22K_b1024_wd0.05_lr1e-3_lrd0.75_pdp0.0_dp0.2_warm5_s0_beta0.999_p1st_stop0_helloworld' \
    # --config.pretrain_fmt=t5x \


    # --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220526_052256_maet5x_kmh-tpuvm-v3-256-1_cfg_mae_large_800ep_b4096_lr1e-4_mk0.75_s100_p1_vis' \
    # --config.pretrain_fmt='t5x' \

    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220521_221137_scratch_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_fttl_b1024_wd0.3_lr1e-4_lrd1.0_dp0.2_warm20_s0_beta0.95_p16_dbgp16/checkpoint_62550'
    # --config.pretrain_dir='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax'
    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220520_203852_scratch_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_fttl_b1024_wd0.3_lr1e-4_lrd1.0_dp0.2_warm20_s0_beta0.95_p1_hwrng_lrd/checkpoint_50040'




