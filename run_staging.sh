VM_NAME=kmh-tpuvm-v3-256-4
echo $VM_NAME
REPO=https://71d519550fe3430ecbf39b70467e9210aed5da69:@github.com/KaimingHe/flax_dev.git
BRANCH=main

# kaiming: this is the setting that produced the crash (v3-256 only)
CONFIG=cfg_vit_large
batch=4096  # 4096 for 256 TPUs (16 each)
init_backend=tpu
ema=True
donate=True
mu_type=bfloat16
log_every_steps=1

JOBNAME=flax/$(date +%Y%m%d_%H%M%S)_${VM_NAME}_${CONFIG}_${ep}ep_pytorch_recipe_batch${batch}


DATADIR='gs://kmh-gcp/tensorflow_datasets'
WORKDIR='gs://kmh-gcp/checkpoints/'${JOBNAME}
LOGDIR=/home/${USER}/logs/${JOBNAME}
mkdir -p ${LOGDIR}

# source run_init_remote.sh

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd ~/flax_dev
git pull
git checkout vit.bug
git rev-parse --short HEAD

# pip3 list | grep 'jax\|flax\|tensorflow '

cd ~/flax_dev
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
python3 main.py \
    --workdir=$WORKDIR \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.log_every_steps=${log_every_steps} \
    --config.num_epochs=200 \
    --config.ema_decay=0.9999 \
    --config.ema=${ema} \
    --config.save_every_epochs=10 \
    --config.opt_mu_dtype=${mu_type} \
    --config.donate=${donate} \
    --config.init_backend=${init_backend} \
" 2>&1 | tee $LOGDIR/finetune.log

echo ${VM_NAME}
