rm -rf tmp

# 4096 / 256 tpus = 128 / 8 tpus  

# export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
# python3 main.py \
#     --workdir=./tmp \
#     --config=configs/cfg_vit_large.py \
#     --config.batch_size=128 \
#     --config.log_every_steps=10 \
#     --config.num_epochs=0.005 \
#     --config.profile_memory=True \
#     --config.ema=True \
#     --config.donate=True \
#     --config.model.classifier='tgap' \
#     --config.pretrain_dir='gs://kmh-gcp/from_pytorch/checkpoint/kaiminghe/converted/2021-10-26-03-09-46-v3-128-mb4096-epo800-PMAEp16-ViTLarge-lr1e-4-wd5e-2-warm40-mask0.75-pred8d512-exNB-msaLNmlpLNeLNpLNkBN0-1view-NOrelpos-abspos-clstoken-qkv-NOlayerscale-LNtgt_convert_pt2jax' \

    # --config.model.transformer.num_layers=12 \
    # --config.model.hidden_size=768 \
    # --config.model.transformer.num_layers=2 \


source run_convert.sh
