#!/bin/bash

echo $$
set -x

################################################################
# arguments
################################################################
JOB_NAME=$1
TPU_NAME=$2
CONFIG=$3
JOB_DIR=$4
WORK_DIR_PRETRAIN=$5
EXTRA_ARGS_COMMON_TAG=$6

array=( $@ )
len=${#array[@]}
EXTRA_ARGS_COMMON=${array[@]:6:$len}

FOLDER=vit_jax
STORAGE_BUCKET=gs://xinleic
DATASET=imagenet-1k
if [ ${EXTRA_ARGS_COMMON_TAG} = "default" ]; then
    TUNE_TAG=finetune
else
    TUNE_TAG=finetune_${EXTRA_ARGS_COMMON_TAG}
fi
################################################################
# folders
################################################################
WORK_DIR=${STORAGE_BUCKET}/checkpoints/${JOB_DIR}/${TUNE_TAG}

LOG_DIR=/checkpoint/$USER/logs/${JOB_DIR}/${TUNE_TAG}
sudo mkdir -p ${LOG_DIR} && sudo chmod -R 777 ${LOG_DIR}

################################################################
# staging
################################################################
TAG_WITH_TIME=${JOB_NAME}_`date +'%Y-%m-%d_%H-%M-%S'`
STAGE_DIR=/checkpoint/$USER/stages/${JOB_DIR}/${FOLDER}_${TAG_WITH_TIME}_${TUNE_TAG}
echo $STAGE_DIR
mkdir -p $STAGE_DIR
rsync -avz $HOME/$FOLDER/ $STAGE_DIR/

# so that it can be sync-ed well?
sleep 5

################################################################
# launch on all nodes
################################################################
cd ${HOME} && gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone europe-west4-a --worker all --command "
cd $STAGE_DIR

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export LOCAL_REDIRECT_CKPT_DIR=${WORK_DIR}

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/cfg_vit_${CONFIG}.py \
    --config.pretrain_dir=${WORK_DIR_PRETRAIN} \
    --config.torchload.data_dir=/datasets/${DATASET} \
    ${EXTRA_ARGS_COMMON} \
    2>&1 | tee $LOG_DIR/finetune_\${SSH_CLIENT// /_}_${TAG_WITH_TIME}.log

if [ \${PIPESTATUS[0]} -eq 0 ]; then
    touch $LOG_DIR/${TUNE_TAG}.flag
fi
" 2>&1 | tee $LOG_DIR/finetune_main_${TAG_WITH_TIME}.log

################################################################
# cleanup on all nodes
################################################################
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone europe-west4-a --worker all --command "
sudo pkill python
sudo lsof -w /dev/accel0 | grep .py | awk '{print \"sudo kill -9 \" \$2}' | sh
sudo rm -f /tmp/libtpu_lockfile
mkdir -p /tmp/tpu_logs && sudo chmod a+w -R /tmp/tpu_logs
"

set +x