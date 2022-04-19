rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_vit_dbg.py \
    --config.batch_size=128 \
    --config.log_every_steps=10 \
    --config.num_epochs=0.005 \
    --config.profile_memory=True \
    --config.ema=False \
    --config.donate=True \
    --config.aug.randerase.on=False \
    --config.aug.randerase.prob=0.25 \
    --config.rescale_head_init=0.001 \
    --config.aug.mix.mixup=True \
    --config.aug.mix.cutmix=True \
    --config.aug.mix.switch_mode=host_batch \
    --config.aug.autoaug=randaugv2 \
    --config.model.transformer.torch_qkv=False \
    --config.aug.torchvision=True \
    --config.aug.mix.torchvision=False \
    --config.eval_only=True \

    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax_finetune/20220418_201436_kmh-tpuvm-v3-256-3_cfg_vit_large_50ep_ftpy_b1024_lr1e-3_lrd0.75_dp0.2_randaugv2_shf512x32_hostbatch_seed0_TorchLoader_DBGbest' \

    # --config.resume_dir='gs://kmh-gcp/checkpoints/flax/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax_finetune/20220418_214713_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_ftpy_b1024_lr1e-3_lrd0.75_dp0.2_TVrandaugv2_shf512x32_hostbatch_seed0_torch171'




    # --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220413_000736_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_lr1e-4_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG_masknoise_qkv'

    # --config.model.transformer.num_layers=12 \
    # --config.model.hidden_size=768 \
    # --config.model.transformer.num_layers=2 \


# source run_convert_j2p.sh
