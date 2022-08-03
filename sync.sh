#!/bin/zsh

DEST_RUN=/checkpoint/xinleic/mae_jax/repo_vit
DEST_DEV=vit_jax

for i in configs t5x utils; do 
    rsync -aiz --delete --partial --progress $i/ devtpux4:$DEST_RUN/$i/
    rsync -aiz --delete --partial --progress $i/ devtpux4:$DEST_DEV/$i/
done

rsync -aiz --delete --partial --progress *.py devtpux4:$DEST_RUN/
rsync -aiz --delete --partial --progress *.py devtpux4:$DEST_DEV/
