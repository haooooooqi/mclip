# select ViT-Large

# converted from TF=>PyTorch
# PRETRAIN_DIR='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-03-09-46-v3-128-mb4096-epo800-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt_convert_pt2jax'
# PRETRAIN_DIR='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-22-16-05-v3-128-mb4096-epo1600-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt-resume3_convert_pt2jax'

# debugging, 800ep/1600ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220331_020430_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220401_111439_kmh-tpuvm-v3-256-4_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1_randuniform_normimpl_cropv3'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220402_065729_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_randuniform_normimpl_cropv3_qkvinit_patchinit'  # good
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220401_210908_kmh-tpuvm-v3-128-2_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_randuniform_normimpl_cropv3_qkvinit_NOpatchinit'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220402_204718_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220402_203809_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2'  # 1600ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220403_202910_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220403_224513_kmh-tpuvm-v3-256-4_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2_hostseed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220403_203139_kmh-tpuvm-v3-128-1_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv3sanity'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220404_170716_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2'  # 1600ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220405_050556_kmh-tpuvm-v3-256-3_cfg_mae_large_800ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_syncbn_fixbug'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220406_195729_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220411_203139_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG_shf320b'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220413_000736_kmh-tpuvm-v3-256-4_cfg_mae_large_1600ep_maeDBG_batch4096_lr1e-4_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG_masknoise_qkv'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220413_030239_kmh-tpuvm-v3-128-1_cfg_mae_large_1600ep_maeDBG_batch4096_lr1e-4_vmap_normpix_sincos_initmaev2_cropv2_donate_olkNN_NOexClsDBG_masknoise_qkv_buf16x1024'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220415_011305_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_lr1.0e-4_vmap_normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_masknoise_qkv_buf16x1024_noavelog_seed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220416_053141_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_lr1.0e-4_vmap_normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_masknoise_qkv_buf16x1024_seed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220417_063852_kmh-tpuvm-v3-256-3_cfg_mae_large_1600ep_maeDBG_batch4096_lr1.0e-4_vmap_normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_buf16x1024_seed'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220421_070850_kmh-tpuvm-v3-256-1_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_staging'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220423_072315_kmh-tpuvm-v3-256-3_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_TorchLoader_wseed100_dbg_save50_resume_sanity'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220423_054117_kmh-tpuvm-v3-256-4_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_TorchLoader_wseed100_dbg_save50'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220422_170904_kmh-tpuvm-v3-256-1_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_staging'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220507_175337_kmh-tpuvm-v3-256-4_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100'  # 800ep

# debugging, 200ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220402_015256_kmh-tpuvm-v3-256-4_cfg_mae_large_200ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev2_randuniform_normimpl_cropv3_qkvinit_patchinit'

# explore: more cls tokens
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220507_232714_kmh-tpuvm-v3-256-1_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_cls15_dbgpos'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220508_184006_kmh-tpuvm-v3-256-1_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_cls79_dbgpos'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220509_045043_kmh-tpuvm-v3-256-4_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_cls147_dbgpos'

# explore: autoreg
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220626_063055_kmh-tpuvm-v3-256-4_cfg_mae_large_autoreg_800ep_b4096_lr1.0e-4_TorchLoader_wseed100_normpix'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220702_192830_kmh-tpuvm-v3-512-1_cfg_mae_large_autoreg_800ep_b4096_lr1.0e-4_TorchLoader_wseed100_normpix_ohem0_off10'
PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220703_013446_kmh-tpuvm-v3-256-2_cfg_mae_large_autoreg_800ep_b4096_lr1.0e-4_TorchLoader_wseed100_normpix_ohem0_off20'

# explore: FFT
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220628_011709_kmh-tpuvm-v3-256-4_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_FFT_2dec'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220628_065825_kmh-tpuvm-v3-256-3_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_FFT_decFSPdct'

# 100/200/400ep MAE baseline
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220630_165012_kmh-tpuvm-v3-256-4_cfg_mae_large_maetf_100ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220702_062025_kmh-tpuvm-v3-256-2_cfg_mae_large_maetf_200ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220702_062042_kmh-tpuvm-v3-256-3_cfg_mae_large_maetf_400ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100'

# explore: layerwise pre-train
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220630_182019_kmh-tpuvm-v3-256-2_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_wseed100_16layers'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220701_081101_kmh-tpuvm-v3-256-2_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_wseed100_24layers'

# Lseed
# gs://kmh-gcp/checkpoints/flax/20220701_102256_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed121_21layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_061006_kmh-tpuvm-v3-256-2_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_wseed100_2layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_143435_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed103_3layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_150406_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed104_4layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_153610_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed105_5layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_161226_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed106_6layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_165255_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed107_7layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_173753_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed108_8layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_182700_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed109_9layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_192044_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed110_10layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_201843_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed111_11layers/checkpoint_31200
# PRETRAIN_DIR="gs://kmh-gcp/checkpoints/flax/20220630_212104_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed112_12layers"
# gs://kmh-gcp/checkpoints/flax/20220630_222800_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed113_13layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220630_233922_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed114_14layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220701_005456_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed115_15layers/checkpoint_31200
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220701_021526_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed116_16layers'
# gs://kmh-gcp/checkpoints/flax/20220701_034151_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed117_17layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220701_051318_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed118_18layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220701_065419_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed119_19layers/checkpoint_31200
# PRETRAIN_DIR="gs://kmh-gcp/checkpoints/flax/20220701_083628_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed120_20layers"
# gs://kmh-gcp/checkpoints/flax/20220701_102256_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed121_21layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220701_121453_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed122_22layers/checkpoint_31200
# gs://kmh-gcp/checkpoints/flax/20220701_141157_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed123_23layers/checkpoint_31200
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220701_161455_kmh-tpuvm-v3-256-3_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_Lseed124_24layers'

# Lseed, 200ep
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220702_064754_kmh-tpuvm-v3-256-4_cfg_mae_large_maeLW_200ep_b4096_lr1.0e-4_mask0.75_Lseed116_16layers'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220703_095146_kmh-tpuvm-v3-256-4_cfg_mae_large_maeLW_200ep_b4096_lr1.0e-4_mask0.75_Lseed124_24layers'