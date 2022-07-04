# source run_env.sh

rm -rf tmp

# 4096 / 128 tpus = 256 / 8 tpus  
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_mae_dbg.py \
    --config.batch_size=256 \
    --config.log_every_steps=100 \
    --config.num_epochs=10 \
    --config.profile_memory=True \
    --config.donate=False \
    --config.model.norm_pix_loss=True \
    --config.model.sincos=False \
    --config.aug.crop_ver=v2 \
    --config.model.visualize=True \
    --config.model.transformer.torch_qkv=False \
    --config.model.transformer.num_layers=2 \
    --config.model.decoder.transformer.num_layers=2 \
    --config.model.freeze_layers=1 \

    # --config.pretrain_dir='gs://kmh-gcp/checkpoints/flax/20220630_035900_kmh-tpuvm-v3-256-2_cfg_mae_large_maeLW_100ep_b4096_lr1.0e-4_mask0.75_wseed100_1layers'

    # --config.model.transformer.num_layers=2 \
    # --config.model.patches.size=\(16,16\) \


# python3 test_profile.py

# pprof -http=localhost:6062 tmp/memory.prof

# PROF_DIR='gs://foo/bar'
# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8` && TGT_DIR='/tmp/'`basename $PROF_DIR`'_memory_'${salt}'.prof' && gsu cp $PROF_DIR/memory.prof $TGT_DIR && echo $TGT_DIR
# pprof -http=localhost:6062 $TGT_DIR
