#!/bin/bash

if [ ! -n "$1" ]; then
    echo "Usage: " $0 " {TPU_NAME}"
    echo "       {TPU_NAME} is MANDATORY"
    exit
fi

TPU_NAME=$1
ZONE=europe-west4-a
PROJECT_ID=fair-infra3f4ebfe6

echo 'To kill jobs in: '${TPU_NAME}' after 2s...'
sleep 2s

echo 'Killing jobs...'
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker all \
  --command "
sudo pkill python
sudo lsof -w /dev/accel0 | grep .py | awk '{print \"sudo kill -9 \" \$2}' | sh
sudo rm -f /tmp/libtpu_lockfile
mkdir -p /tmp/tpu_logs && sudo chmod a+w -R /tmp/tpu_logs
"

echo 'Killed jobs.'
