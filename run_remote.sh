# run remote

lr=0.1e-4  # 1e-4
ep=800
batch=4096
mask=0.75

full=8
drop=0.25

seed=100

vitsize=large
CONFIG=cfg_mae_${vitsize}
source scripts/select_chkpt_${vitsize}.sh

# maetf: normpix_sincos_initmaev2_cropv2ALTER_donate_olkNN_NOexClsDBG_buf16x1024 (torch loader: crop v4)
JOBNAME=flax/$(date +%Y%m%d_%H%M%S)_${VM_NAME}_${CONFIG}_maetf_${ep}ep_b${batch}_lr${lr}_mask${mask}_TorchLoader_wseed${seed}_full${full}_drop${drop}
RESUME_DIR=''

WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/kmh_data/logs/${JOBNAME}
mkdir -p ${LOGDIR}
chmod 777 ${LOGDIR}

# source run_init_remote.sh

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd $STAGEDIR
git config --global --add safe.directory $STAGEDIR

echo Current commit: $(git show -s --format=%h)

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

source run_get_ssh_id.sh

python3 main.py \
    --workdir=$WORKDIR \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.save_every_epochs=50 \
    --config.model.norm_pix_loss=True \
    --config.model.sincos=True \
    --config.model.mask_ratio=${mask} \
    --config.aug.crop_ver=v2 \
    --config.donate=True \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.seed_tf=${seed} \
    --config.resume_dir=$RESUME_DIR \
    --config.pretrain_dir=$PRETRAIN_DIR \
    --config.model.full_blocks=${full} \
    --config.model.drop_channels=${drop} \
2>&1 | tee $LOGDIR/pretrain_\$SSH_ID.log
" 2>&1 | tee $LOGDIR/pretrain.log


echo ${VM_NAME}
