# select ViT-Base

# debugging, 100ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220331_014514_kmh-tpuvm-v3-128-2_cfg_mae_base_100ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220331_030004_kmh-tpuvm-v3-128-2_cfg_mae_base_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1'

# supervised MAE
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220821_201849_kmh-tpuvm-v3-256-3_cfg_mae_base_maetf_400ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_sup'
PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220821_202455_kmh-tpuvm-v3-256-7_cfg_mae_base_maetf_400ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_sup_dec1'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220821_205858_kmh-tpuvm-v3-256-4_cfg_mae_base_maetf_400ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_sup_dec1_w0.01'
