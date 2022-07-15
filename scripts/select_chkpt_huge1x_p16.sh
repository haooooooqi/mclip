# select ViT-Huge1x/16

# helloworld
# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220626_031329_maet5x_kmh-tpuvm-v3-256-3_cfg_mae_huge1x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p1_re1.0_normpix_exwd_NOsplit_fastsave'

# IN22k FT
PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220626_031329_maet5x_kmh-tpuvm-v3-256-3_cfg_mae_huge1x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p1_re1.0_normpix_exwd_NOsplit_fastsave_finetune/20220712_193607_kmh-tpuvm-v3-256-6_cfg_vit_huge1x_p16_50ep_fttl_IN22K_b1024_wd0.05_lr1e-3_lrd0.75_pdp0.0_dp0.2_warm5_s0_beta0.999_p1st_stop0_act1_param1'