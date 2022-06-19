CODEDIR=/checkpoint/xinleic/mae_jax/repo

TPU_NAME=xinleic-mae-iv-1
ZONE=europe-west4-a

################################################################
# configs
################################################################

batch=4096
lr=1.5e-4
ep=1600
mask=0.75
rescale=1.0
vitsize=huge

seed=0
partitions=1

CONFIG=cfg_mae_${vitsize}
JOBNAME=huge_p14_v1

WORKDIR=gs://xinleic/mae_jax/checkpoints/${JOBNAME}
RESUME_DIR=''
LOGDIR=/checkpoint/xinleic/mae_jax/logs/${JOBNAME}
sudo mkdir -p ${LOGDIR} && sudo chmod -R 777 ${LOGDIR}

################################################################
# launch on all nodes
################################################################

cd ${HOME} && gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker all \
  --command "
cd $CODEDIR

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export LOCAL_REDIRECT_CKPT_DIR=${WORKDIR}

python3 main.py \
    --workdir=${LOGDIR} \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.model.transformer.rescale_init=${rescale} \
    --config.profile_memory=False \
    --config.model.norm_pix_loss=True \
    --config.model.sincos=True \
    --config.model.mask_ratio=${mask} \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.partitioning.num_partitions=${partitions} \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
    --config.partitioning.partition_states=False \
    --config.model.visualize=False \
    --config.resume_dir=$RESUME_DIR \
    --config.torchload.data_dir=/datasets/imagenet-1k \
    2>&1 | tee $LOGDIR/pretrain_\${SSH_CLIENT// /_}.log
" 2>&1 | tee -a $LOGDIR/pretrain.log