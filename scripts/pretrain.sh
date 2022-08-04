#!/bin/bash

echo $$
set -x
JOB_NAME=$1
TPU_NAME=$2
CONFIG=$3
DATASET=$4

################################################################
# arguments
################################################################
array=( $@ )
len=${#array[@]}
# find the index of -- which separates options that are specific to train, and options common to fine-tune
cnt=0; for el in "${array[@]}"; do
    [[ $el == "--" ]] && break
    ((++cnt))
done
echo $cnt

EXTRA_ARGS=${array[@]:4:$len}
if [ $len -eq 4 ]; then
    echo "No option."
    declare EXTRA_ARGS_ALL=()
    declare EXTRA_ARGS_COMMON=()
elif [ $cnt -eq $len ]; then
    echo "All the options are train-specific."
    EXTRA_ARGS_ALL=${array[@]:4:$len}
    declare EXTRA_ARGS_COMMON=()
else
    echo "Training specific options"
    let "train_len = $cnt - 4"
    echo $train_len
    EXTRA_ARGS_ALL=${array[@]:4:$train_len}
    EXTRA_ARGS_COMMON=${array[@]:$cnt+1:$len}
    EXTRA_ARGS_ALL+=${array[@]:$cnt+1:$len}
fi

if [ ${#EXTRA_ARGS_ALL[@]} -gt 0 ]; then
    EXTRA_ARGS_ALL_TAG=`echo ${EXTRA_ARGS_ALL//config./} | sed -e 's/--/%/g' | sed -e 's/=/@/g' | tr -d ' ' `
    EXTRA_ARGS_ALL_TAG=${EXTRA_ARGS_ALL_TAG:1}
else
    EXTRA_ARGS_ALL_TAG=default
fi

if [ ${#EXTRA_ARGS_COMMON[@]} -gt 0 ]; then
    EXTRA_ARGS_COMMON_TAG=`echo ${EXTRA_ARGS_COMMON//config./} | sed -e 's/--/%/g' | sed -e 's/=/@/g' | tr -d ' ' `
    EXTRA_ARGS_COMMON_TAG=${EXTRA_ARGS_COMMON_TAG:1}
else
    EXTRA_ARGS_COMMON_TAG=default
fi

################################################################
# folders
################################################################
FOLDER=mae_jax
TAG_WITH_TIME=${JOB_NAME}_`date +'%Y-%m-%d_%H-%M'`
cd $HOME/$FOLDER
JOB_DIR=$FOLDER/${CONFIG}/${DATASET}/${EXTRA_ARGS_ALL_TAG}
STORAGE_BUCKET=gs://xinleic

WORK_DIR=${STORAGE_BUCKET}/checkpoints/${JOB_DIR}
LOG_DIR=/checkpoint/$USER/logs/${JOB_DIR}
sudo mkdir -p ${LOG_DIR} && sudo chmod -R 777 ${LOG_DIR}

################################################################
# staging
################################################################
STAGE_DIR=/checkpoint/$USER/stages/${JOB_DIR}/${TAG_WITH_TIME}
echo $STAGE_DIR
mkdir -p $STAGE_DIR
rsync -avz $HOME/$FOLDER/ $STAGE_DIR/

# so that it can be sync-ed well?
sleep 5

################################################################
# launch on all nodes
################################################################
cd ${HOME} && gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone europe-west4-a --worker all \
  --command "
cd $STAGE_DIR

export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export LOCAL_REDIRECT_CKPT_DIR=${WORK_DIR}

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/cfg_mae_${CONFIG}.py \
    --config.resume_dir='' \
    --config.torchload.data_dir=/datasets/${DATASET} \
    ${EXTRA_ARGS_ALL} \
    2>&1 | tee $LOG_DIR/pretrain_\${SSH_CLIENT// /_}_${TAG_WITH_TIME}.log
" 2>&1 | tee $LOG_DIR/pretrain_${TAG_WITH_TIME}.log

PRETRAIN_STATUS=${PIPESTATUS[0]}

################################################################
# launch on all nodes
################################################################
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone europe-west4-a --worker all \
  --command "
sudo pkill python
sudo lsof -w /dev/accel0 | grep .py | awk '{print \"sudo kill -9 \" \$2}' | sh
sudo rm -f /tmp/libtpu_lockfile
mkdir -p /tmp/tpu_logs && sudo chmod a+w -R /tmp/tpu_logs
"

# if [ $PRETRAIN_STATUS -eq 0 ]; then
#     touch $LOG_DIR/pretrain.flag
#     bash finetune_npe.sh $JOB_NAME $TPU_NAME $NUM_CORE $CONFIG $LOG_DIR $MODEL_DIR $EXTRA_ARGS_COMMON
#     touch $LOG_DIR/done.flag
# fi

set +x