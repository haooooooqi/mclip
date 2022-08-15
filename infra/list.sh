#!/bin/bash

if [ -e $1 ]; then
    echo "bash jobs:"
    ps aux | grep "/home/xinleic/mae_jax/scripts/\|/home/xinleic/vit_jax/scripts/" | grep "/bin/bash"
else
    echo "python3 jobs:"
    ps aux | grep "google-cloud-sdk/lib/gcloud.py alpha compute tpus" | grep "python3"
fi
