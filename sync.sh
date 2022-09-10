#!/bin/zsh

DEST_DEV=mae_jax

sync_tpu () {
    MACHINE=$1

    rsync -aiz --delete --partial --progress *.py $MACHINE:$DEST_DEV/

    for i in configs scripts infra t5x utils; do
        rsync -aiz --delete --partial --progress $i/ $MACHINE:$DEST_DEV/$i/
    done
}

sync_tpu devtpux4
# sync_tpu devtpuv0