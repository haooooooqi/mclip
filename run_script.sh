rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_vit_large.py \
    --config.batch_size=128 \
    --config.log_every_steps=10 \
    --config.num_epochs=1 \
    --config.profile_memory=True \
    --config.ema=True \
    --config.ema_eval=True \
    --config.donate=True \
    --config.aug.randerase.on=True \
    --config.aug.randerase.prob=0.25 \
    --config.rescale_head_init=0.001 \
    --config.aug.mix.mixup=True \
    --config.aug.mix.cutmix=True \
    --config.aug.autoaug=autoaug \
    --config.model.transformer.torch_qkv=False \
    --config.eval_only=False \
    --config.model.classifier=gap \
    --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220915_022850_kmh-tpuvm-v3-256-4_cfg_mae_large_autoreg_800ep_b4096_lr1.0e-4_TorchLoader_wseed100_normpix_ohem0_off0_dec8_fps'

    # --config.aug.area_range=0.9,1 \
    # --config.aug.aspect_ratio_range=0.8,1.2 \

    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax_finetune/20220418_214713_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_ftpy_b1024_lr1e-3_lrd0.75_dp0.2_TVrandaugv2_shf512x32_hostbatch_seed0_torch171'

    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax_finetune/20220418_201436_kmh-tpuvm-v3-256-3_cfg_vit_large_50ep_ftpy_b1024_lr1e-3_lrd0.75_dp0.2_randaugv2_shf512x32_hostbatch_seed0_TorchLoader_DBGbest' \

    # --config.model.transformer.num_layers=12 \
    # --config.model.hidden_size=768 \
    # --config.model.transformer.num_layers=2 \


# source run_convert_j2p.sh
