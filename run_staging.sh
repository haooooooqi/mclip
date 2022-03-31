# VM_NAME=kmh-tpuvm-v3-128-2
VM_NAME=kmh-tpuvm-v3-256-4
echo $VM_NAME
REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main

# salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`

ep=100
batch=4096


CONFIG=cfg_vit_base

PRETRAIN_DIR='gs://kmh-gcp/checkpoints/flax/20220331_014514_kmh-tpuvm-v3-128-2_cfg_mae_base_100ep_maeDBG_batch4096_vmap_normpix_sincos_initmaev1'
name=`basename ${PRETRAIN_DIR}`

# pytorch_recipe: _autoaug_lb0.1_cropv4_exwd_initv2_rsinit_dp0.1_cutmixup_minlr
JOBNAME=flax/${name}_finetune/$(date +%Y%m%d_%H%M%S)_${VM_NAME}_${CONFIG}_${ep}ep_pytorch_recipe_b${batch}

WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/home/${USER}/logs/${JOBNAME}
mkdir -p ${LOGDIR}

# source run_init_remote.sh

# check libraries
# gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
#     --worker=0 --command "
# pip3 list | grep jax
# pip3 list | grep flax
# pip3 list | grep tensorflow
# "

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd ~/flax_dev
git pull
git checkout vit.ft
git pull
git rev-parse --short HEAD

# pip3 list | grep 'jax\|flax\|tensorflow '

cd ~/flax_dev
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=$WORKDIR \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.ema=True \
    --config.save_every_epochs=10 \
    --config.profile_memory=True \
    --config.donate=True \
    --config.init_backend=tpu \
    --config.pretrain_dir=${PRETRAIN_DIR} \
" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}
