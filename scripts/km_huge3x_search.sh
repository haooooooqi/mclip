CODEDIR=/checkpoint/xinleic/mae_jax/repo_vit

TPU_NAME=xinleic-mae-iv-0
ZONE=europe-west4-a

################################################################
# configs
################################################################

vitsize=huge3x
batch=1024
lr=1e-3
wd=0.03
lrd=0.75
ep=50
warm=5
dp=0.3
beta2=0.999

seed=0
partitions=8
partition_states=True

CONFIG=cfg_vit_${vitsize}
JOBNAME=${vitsize}

PRETRAIN_DIR=gs://kmh-gcp/checkpoints/flax/20220606_070525_maet5x_kmh-tpuvm-v3-256-3_cfg_mae_huge3x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p4_re1.0_normpix_exwd_split_fastsave
WORKDIR=gs://xinleic/mae_jax/checkpoints/km_tune/${JOBNAME}/wd@${wd}
LOGDIR=/checkpoint/xinleic/mae_jax/logs/km_tune/${JOBNAME}/wd@${wd}
sudo mkdir -p ${LOGDIR} && sudo chmod -R 777 ${LOGDIR}

################################################################
# launch on all nodes
################################################################

cd ${HOME} && gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker=all \
  --command "
cd $CODEDIR

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export LOCAL_REDIRECT_CKPT_DIR=${WORKDIR}

python3 main.py \
    --workdir=${LOGDIR} \
    --config=configs/$CONFIG.py \
    --config.pretrain_dir=${PRETRAIN_DIR} \
    --config.batch_size=${batch} \
    --config.log_every_steps=100 \
    --config.learning_rate=${lr} \
    --config.learning_rate_decay=${lrd} \
    --config.opt.weight_decay=${wd} \
    --config.opt.b2=${beta2} \
    --config.warmup_epochs=${warm} \
    --config.num_epochs=${ep} \
    --config.save_every_epochs=50 \
    --config.profile_memory=False \
    --config.donate=True \
    --config.init_backend=tpu \
    --config.aug.mix.mixup=True \
    --config.aug.mix.cutmix=True \
    --config.aug.randerase.on=True \
    --config.aug.autoaug=randaugv2 \
    --config.model.transformer.droppath_rate=${dp} \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.model.classifier=tgap \
    --config.partitioning.num_partitions=${partitions} \
    --config.pretrain_fmt=t5x \
    --config.partitioning.force_partition_states_data_first=${partition_states} \
    --config.partitioning.partition_states=${partition_states} \
    --config.torchload.data_dir=/datasets/imagenet-1k \
    2>&1 | tee -a $LOGDIR/pretrain_\${SSH_CLIENT// /_}.log
" 2>&1 | tee -a $LOGDIR/finetune.log

# kill all the jobs
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --worker all \
  --command "
sudo pkill python
sudo lsof -w /dev/accel0 | grep .py | awk '{print \"sudo kill -9 \" \$2}' | sh
sudo rm -f /tmp/libtpu_lockfile
mkdir -p /tmp/tpu_logs && sudo chmod a+w -R /tmp/tpu_logs
"