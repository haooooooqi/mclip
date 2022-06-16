#!/bin/zsh

DEST_RUN=/checkpoint/xinleic/mae_jax/repo
DEST_DEV=mae_jax

for i in configs scripts t5x utils; do 
    rsync -aiz --delete --partial --progress $i/ devtpuv0:$DEST_RUN/$i/
    rsync -aiz --delete --partial --progress $i/ devtpuv0:$DEST_DEV/$i/
done

rsync -aiz --delete --partial --progress *.py devtpuv0:$DEST_RUN/
rsync -aiz --delete --partial --progress *.py devtpuv0:$DEST_DEV/
