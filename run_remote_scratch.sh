echo 'code dir: '$STAGEDIR

# seed=0
batch=1024
lr=1e-4
wd=0.3
lrd=1.0
ep=50
warm=20
dp=0.2
beta2=0.95

partitions=1

vitsize=large
CONFIG=cfg_vit_${vitsize}

# source scripts/select_chkpt_${vitsize}.sh
# name=`basename ${PRETRAIN_DIR}`
# --config.pretrain_dir=${PRETRAIN_DIR} \

# finetune_pytorch_recipe (ftpy): lb0.1_b0.999_cropv4_exwd_initv2_headinit0.001_tgap_dp_mixup32_cutmix32_noerase_warmlr_minlr_autoaug
# finetune_torch_loader (fttl): randaugv2erase_TorchLoader
JOBNAME=flax/$(date +%Y%m%d_%H%M%S)_scratch_${VM_NAME}_${CONFIG}_${ep}ep_fttl_b${batch}_wd${wd}_lr${lr}_lrd${lrd}_dp${dp}_warm${warm}_s${seed}_beta${beta2}_p${partitions}_hwrng_save1

# RESUME_DIR='gs://kmh-gcp/checkpoints/flax/20220520_203852_scratch_kmh-tpuvm-v3-256-1_cfg_vit_large_50ep_fttl_b1024_wd0.3_lr1e-4_lrd1.0_dp0.2_warm20_s0_beta0.95_p1_hwrng_lrd/checkpoint_50040'
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
echo Current dir: $(pwd)

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

source run_get_ssh_id.sh

python3 main.py \
    --workdir=${WORKDIR} \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.learning_rate=${lr} \
    --config.learning_rate_decay=${lrd} \
    --config.opt.weight_decay=${wd} \
    --config.opt.b2=${beta2} \
    --config.warmup_epochs=${warm} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.save_every_epochs=1 \
    --config.profile_memory=True \
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
    --config.model.transformer.torch_qkv=False \
    --config.model.classifier=token \
    --config.partitioning.num_partitions=${partitions} \
    --config.resume_dir=${RESUME_DIR} \
2>&1 | tee $LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}