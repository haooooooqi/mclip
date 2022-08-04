#!/bin/bash

SCRIPT_NAME=$1
JOB_NAME=$2
NUM_CORE=$3
array=( $@ )
len=${#array[@]}

EXTRA_ARGS=${array[@]:3:$len}

LOG_DIR=${HOME}/logs
if [ ! -d ${LOG_DIR} ]; then
    mkdir -p $LOG_DIR
fi

LOG_PREFIX="${LOG_DIR}/`date +'%Y-%m-%d_%H-%M-%S'`_$$_${JOB_NAME}"

SUBMITTED=0
if [ $NUM_CORE -eq 128 ]; then
    for TPU_NAME in xinleic-mae-i-0 xinleic-mae-i-1 xinleic-mae-i-2 xinleic-mae-i-3; do
        TPU_IN_USE=`~/mae_jax/infra/list.sh | grep $TPU_NAME`
        if [ -z "${TPU_IN_USE}" ]; then
            nohup $HOME/mae_jax/scripts/${SCRIPT_NAME}.sh $JOB_NAME $TPU_NAME ${EXTRA_ARGS} 1>${LOG_PREFIX}.out 2>${LOG_PREFIX}.err &
            SUBMITTED=1
            break
        fi
    done
elif [ $NUM_CORE -eq 256 ]; then
    for TPU_NAME in xinleic-mae-ii-0 xinleic-mae-ii-1 xinleic-mae-ii-2 xinleic-mae-ii-3; do
        TPU_IN_USE=`~/mae_jax/infra/list.sh | grep $TPU_NAME`
        if [ -z "${TPU_IN_USE}" ]; then
            nohup $HOME/mae_jax/scripts/${SCRIPT_NAME}.sh $JOB_NAME $TPU_NAME ${EXTRA_ARGS} 1>${LOG_PREFIX}.out 2>${LOG_PREFIX}.err &
            SUBMITTED=1
            break
        fi
    done
elif [ $NUM_CORE -eq 512 ]; then
    for TPU_NAME in xinleic-mae-iv-0 xinleic-mae-iv-1 xinleic-mae-iv-2 xinleic-mae-iv-3; do
        TPU_IN_USE=`~/mae_jax/infra/list.sh | grep $TPU_NAME`
        if [ -z "${TPU_IN_USE}" ]; then
            nohup $HOME/mae_jax/scripts/${SCRIPT_NAME}.sh $JOB_NAME $TPU_NAME ${EXTRA_ARGS} 1>${LOG_PREFIX}.out 2>${LOG_PREFIX}.err &
            SUBMITTED=1
            break
        fi
    done
fi
