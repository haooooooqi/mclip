# select ViT-Huge4x/16

# helloworld
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220529_035953_maet5x_kmh-tpuvm-v3-512-1_cfg_mae_huge4x_p16_800ep_b4096_lr1e-4_mk0.75_s100_p8_re0.5_normpix_exwd_split_fastsave'
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220606_004214_maet5x_kmh-tpuvm-v3-512-2_cfg_mae_huge4x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p8_re0.5_normpix_exwd_split_fastsave'

# IN22k FT
PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220606_004214_maet5x_kmh-tpuvm-v3-512-2_cfg_mae_huge4x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p8_re0.5_normpix_exwd_split_fastsave_finetune/20220711_220339_kmh-tpuvm-v3-512-1_cfg_vit_huge4x_p16_50ep_fttl_IN22K_b1024_wd0.05_lr1e-3_lrd0.75_pdp0.0_dp0.2_warm5_s4_beta0.999_p4st_stop16_act2_param2'