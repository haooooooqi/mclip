# source run_env.sh

rm -rf tmp

# 4096 / 128 tpus = 256 / 8 tpus  
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=./tmp \
    --config=configs/cfg_mae_base.py \
    --config.batch_size=128 \
    --config.log_every_steps=100 \
    --config.num_epochs=10 \
    --config.profile_memory=True \
    --config.donate=False \
    --config.model.norm_pix_loss=True \
    --config.resume_dir='gs://kmh-gcp/checkpoints/flax/20220331_014514_kmh-tpuvm-v3-128-2_cfg_mae_base_100ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1/checkpoint_31200' \
    --config.model.sincos=False

    # --config.model.visualize=True \

    # --config.model.transformer.num_layers=2 \
    # --config.model.patches.size=\(16,16\) \


# python3 test_profile.py

# pprof -http=localhost:6062 tmp/memory.prof

# PROF_DIR='gs://foo/bar'
# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8` && TGT_DIR='/tmp/'`basename $PROF_DIR`'_memory_'${salt}'.prof' && gsu cp $PROF_DIR/memory.prof $TGT_DIR && echo $TGT_DIR
# pprof -http=localhost:6062 $TGT_DIR
