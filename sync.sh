#!/bin/zsh

DEST_DEV=vit_jax

for i in configs scripts t5x utils; do 
    rsync -aiz --delete --partial --progress $i/ devtpux4:$DEST_DEV/$i/
done

rsync -aiz --delete --partial --progress *.py devtpux4:$DEST_DEV/