# select ViT-Base

# debugging, 100ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220331_014514_kmh-tpuvm-v3-128-2_cfg_mae_base_100ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220331_030004_kmh-tpuvm-v3-128-2_cfg_mae_base_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1'

# SimCLR
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220810_060227_kmh-tpuvm-v3-256-3_cfg_clr_base_simclr_300ep_b4096_lr1e-4_wd0.1_TorchLoader_wseed100_t0.2'
PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220811_060213_kmh-tpuvm-v3-256-3_cfg_clr_base_asymclr_300ep_b4096_lr1e-4_wd0.1_TorchLoader_wseed100_t0.2'