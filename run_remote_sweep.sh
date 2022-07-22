echo 'code dir: '$STAGEDIR

# seed=0
batch=1024
lr=$1
lrd=$2
ep=$3
dp=$4
wep=$5
ema=0.9999
mlr=2.5e-7
wlr=2.5e-7

vitsize=base
CONFIG=cfg_vit_${vitsize}
source scripts/select_chkpt_${vitsize}.sh

PRETRAIN_DIR='gs://shoubhikdn_storage/checkpoints/flax/mae_convnext_base/20220721_091038_cx_256b_cfg_mae_base_maetf_100ep_10wep_b4096_lr0.5e-4_mlr0.25e-5_wlr0.25e-6_psize32_lrscosine_TorchLoader_wseed100_simmim'
name=`basename ${PRETRAIN_DIR}`

# finetune_pytorch_recipe (ftpy): lb0.1_b0.999_cropv4_exwd_initv2_headinit0.001_tgap_dp_mixup32_cutmix32_noerase_warmlr_minlr_autoaug
# finetune_torch_loader (fttl): randaugv2erase_TorchLoader
JOBNAME=flax_dev_ft/mae_convnext_${vitsize}_sweep/${name}_finetune/$(date +%Y%m%d_%H%M%S)_${VM_NAME}_${CONFIG}_${ep}ep_${wep}wep_fttl_b${batch}_lr${lr}_lrd${lrd}_dp${dp}_mlr${mlr}_wlr${wlr}_s${seed}_${ema}

WORKDIR=gs://shoubhikdn_storage/checkpoints/${JOBNAME}
LOGDIR=/home/${USER}/logs/${JOBNAME}
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
    --config.pretrain_dir=${PRETRAIN_DIR} \
    --config.batch_size=${batch} \
    --config.learning_rate=${lr} \
    --config.learning_rate_decay=${lrd} \
    --config.min_abs_lr=${mlr} \
    --config.warmup_abs_lr=${wlr} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.warmup_epochs=${wep} \
    --config.save_every_epochs=10 \
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
    --config.ema=True \
    --config.ema_eval=True \
    --config.ema_decay=${ema} \
    --config.model.classifier=token \
2>&1 | tee $LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}