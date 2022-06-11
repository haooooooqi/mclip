#!/bin/zsh

DEST=/checkpoint/xinleic/mae_jax/repo

for i in configs scripts t5x utils; do 
    rsync -aiz --delete --partial --progress $i/ devtpuv0:$DEST/$i/
    rsync -aiz --delete --partial --progress $i/ devtpuv0:mae_jax/$i/
done

rsync -aiz --delete --partial --progress *.py devtpuv0:$DEST/
rsync -aiz --delete --partial --progress *.py devtpuv0:mae_jax/
