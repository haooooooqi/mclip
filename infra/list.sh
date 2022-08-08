#!/bin/bash

if [ -e $1 ]; then
    echo "bash jobs:"
    ps aux | grep "pretrain\.sh\|finetune\.sh" | grep "/bin/bash"
else
    echo "python3 jobs:"
    ps aux | grep "google-cloud-sdk/lib/gcloud.py alpha compute tpus" | grep "python3"
fi
