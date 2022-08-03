#!/bin/bash

clean_up_tpu() {
    echo "Delete TPU: " $1
    bash tpu_down.sh $1
    echo "Delete TPU: " $2
    bash tpu_down.sh $2
    exit 1
}

echo $$
set -x
JOB_NAME=$1
SCRIPT_PID=$2
NUM_CORE=$3
YAML=$4
STORAGE_BUCKET=gs://xinleic
DATA_DIR=${STORAGE_BUCKET}/in1k
TPU_NAME=$SCRIPT_PID
TPU_EVAL_NAME=ts-$SCRIPT_PID
trap "clean_up_tpu $TPU_NAME $TPU_EVAL_NAME" SIGINT SIGTERM SIGKILL
ACCELERATOR_TYPE=v3-$NUM_CORE
array=( $@ )
len=${#array[@]}
# find the index of -- which separates options that are specific to train, and options common to fine-tune
cnt=0; for el in "${array[@]}"; do
    [[ $el == "--" ]] && break
    ((++cnt))
done
echo $cnt

EXTRA_ARGS=${array[@]:4:$len}
if [ $cnt -eq $len ]; then
    echo "All the options are train-specific."
    EXTRA_ARGS_TRAIN=${array[@]:4:$len}
    EXTRA_ARGS_COMMON=
else
    echo "Training specific options"
    let "train_len = $cnt - 4"
    echo $train_len
    EXTRA_ARGS_TRAIN=${array[@]:4:$train_len}
    EXTRA_ARGS_COMMON=${array[@]:$cnt+1:$len}
fi
cd $HOME/mask/

DRY_RUN_OUT=`python3 -u main.py --config_file=configs/main/${YAML}.yaml ${EXTRA_ARGS_TRAIN} ${EXTRA_ARGS_COMMON}`

if [ $? -ne 0 ]; then
    exit 1
fi
FOLDER=mask
JOB_DIR=$FOLDER/${YAML}/IN1K/${DRY_RUN_OUT}
MODEL_DIR=${STORAGE_BUCKET}/checkpoints/${JOB_DIR}
LOG_DIR=$HOME/logs/${JOB_DIR}
mkdir -p $LOG_DIR

now=`date +'%Y-%m-%d_%H-%M-%S'`
STAGE_DIR=$HOME/stages/${JOB_DIR}_${now}
echo $STAGE_DIR
mkdir -p $STAGE_DIR
rsync -avz $HOME/mask/ $STAGE_DIR/
cd $STAGE_DIR

DELAY=$(( $RANDOM % 10 ))
sleep $DELAY
bash tpu_up.sh $TPU_NAME $ACCELERATOR_TYPE

python3 -u main.py --config_file=configs/main/${YAML}.yaml \
    --salt=${JOB_NAME} \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --tpu=${TPU_NAME} \
    --num_cores=${NUM_CORE} \
    --mode=train \
    --skip_host_call=False \
    --dry_run=False \
    --num_train_images=1281167 \
    --train_epochs=800 \
    --warmup_epochs=40 \
    ${EXTRA_ARGS_TRAIN} ${EXTRA_ARGS_COMMON} \
2>&1 | tee $LOG_DIR/pretrain_${now}.log

RETURN_STATUS=${PIPESTATUS[0]}

if [ $RETURN_STATUS -eq 0 ]; then
    touch $LOG_DIR/pretrain.flag
    bash finetune_npe.sh $JOB_NAME $TPU_NAME $NUM_CORE $YAML $LOG_DIR $MODEL_DIR $TPU_EVAL_NAME $EXTRA_ARGS_COMMON
    # bash finetune_dser.sh $JOB_NAME $TPU_NAME $NUM_CORE $YAML $LOG_DIR $MODEL_DIR $TPU_EVAL_NAME $EXTRA_ARGS_COMMON
    touch $LOG_DIR/done.flag
    bash tpu_down.sh $TPU_NAME
elif [ $RETURN_STATUS -eq 130 ]; then
    bash tpu_down.sh $TPU_NAME
    echo "Submitting the current job since it receives TPU error"
    queue_file=$HOME/tpus/queue.txt
    lock_dir=$HOME/tpus/lock
    mkdir -p $lock_dir
    # tmp_file=${queue_file}.${TPU_NAME}.${now}
    # echo "~/mask/wrapper.sh train $JOB_NAME $NUM_CORE $YAML $EXTRA_ARGS" > $tmp_file
    echo "~/mask/wrapper.sh train $JOB_NAME $NUM_CORE $YAML $EXTRA_ARGS" >> $queue_file
    rmdir $lock_dir
else
    bash tpu_down.sh $TPU_NAME
fi

set +x