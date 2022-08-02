[1mdiff --git a/run_staging.sh b/run_staging.sh[m
[1mindex 7d9c1894..120e35f6 100644[m
[1m--- a/run_staging.sh[m
[1m+++ b/run_staging.sh[m
[36m@@ -1,5 +1,5 @@[m
[31m-VM_NAME=kmh-tpuvm-v3-512-1[m
[31m-# VM_NAME=kmh-tpuvm-v3-256-3[m
[32m+[m[32m# VM_NAME=kmh-tpuvm-v3-512-1[m
[32m+[m[32mVM_NAME=kmh-tpuvm-v3-256-3[m
 echo $VM_NAME[m
 [m
 # ------------------------------------------------[m
[1mdiff --git a/scripts/select_chkpt_large.sh b/scripts/select_chkpt_large.sh[m
[1mindex c944d6c8..346634a8 100644[m
[1m--- a/scripts/select_chkpt_large.sh[m
[1m+++ b/scripts/select_chkpt_large.sh[m
[36m@@ -105,7 +105,7 @@[m
 # PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220726_014123_kmh-tpuvm-v3-256-3_cfg_mae_large_maeclr_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_t0.2_lw1_r0.25_knnclr'[m
 # PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220726_070056_kmh-tpuvm-v3-256-6_cfg_mae_large_maeclr_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_t0.2_lw1_r0.25_un4_knnclr'[m
 [m
[31m-# explore: patch clr[m
[32m+[m[32m# explore: patch clr (bugged)[m
 # PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220730_193810_kmh-tpuvm-v3-256-3_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_tokenv0'[m
 # PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220731_030051_kmh-tpuvm-v3-256-6_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_tokenv0_v2clrt0.2'[m
 # PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220731_080627_kmh-tpuvm-v3-512-1_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_tokenv0_v2clrt0.2_dec4'[m
[36m@@ -113,4 +113,7 @@[m
 # PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220731_075258_kmh-tpuvm-v3-256-7_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_tokenv0_v1clrt0.2'[m
 # PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220731_203826_kmh-tpuvm-v3-256-3_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_tokenv0_v2clrt0.2_dec8_addpix'[m
 # 1600ep[m
[31m-PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220731_202846_kmh-tpuvm-v3-512-1_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_tokenv0_v2clrt0.2_dec8'[m
\ No newline at end of file[m
[32m+[m[32m# PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220731_202846_kmh-tpuvm-v3-512-1_cfg_mae_large_maetf_1600ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_tokenv0_v2clrt0.2_dec8'[m
[32m+[m
[32m+[m[32m# explore: patch clr (pretrain loaded)[m
[32m+[m[32mPRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220802_024910_kmh-tpuvm-v3-256-3_cfg_mae_large_maetf_800ep_b4096_lr1.0e-4_mask0.75_TorchLoader_wseed100_tokenv1_v2clrt0.2_dec8'[m
\ No newline at end of file[m
