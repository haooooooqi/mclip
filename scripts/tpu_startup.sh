rm -f /home/xinleic/.VM_READY

# install software
sudo pip3 install \
  absl-py==1.0.0 \
  flax==0.3.6 \
  jax[tpu]==0.2.21 --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  jaxlib==0.1.71 \
  ml-collections==0.1.0 \
  optax==0.1.0 \
  tensorflow-datasets==4.5.2
sudo pip3 install clu==0.0.7
sudo pip3 uninstall -y tensorflow
sudo pip3 install tf-nightly==2.10.0.dev20220521
sudo pip3 install tensorflow_addons
sudo pip3 install jax[tpu]==0.3.13 jaxlib==0.3.10 flax==0.5.0 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
sudo pip3 install absl-py cached_property gin-config numpy orbax seqio-nightly tensorstore timm

# checkpoint
SHARED_FS=10.89.225.82:/mmf_megavlt
MOUNT_POINT=/checkpoint
for i in $(seq 10); do
  ALREADY_MOUNTED=$(($(df -h | grep $SHARED_FS | wc -l) >= 1))
  if [[ $ALREADY_MOUNTED -ne 1 ]]; then
    sudo apt-get -y update
    sudo apt-get -y install nfs-common
    sudo mkdir -p $MOUNT_POINT
    sudo mount $SHARED_FS $MOUNT_POINT
    sudo chmod go+rw $MOUNT_POINT
  else
    break
  fi
done

# datasets
PD_DEVICE=/dev/sdb
PD_MOUNT_POINT=/datasets
for i in $(seq 10); do
  ALREADY_MOUNTED=$(($(df -h | grep $PD_MOUNT_POINT | wc -l) >= 1))
  if [[ $ALREADY_MOUNTED -ne 1 ]]; then
    sudo mkdir -p $PD_MOUNT_POINT
    sudo mount -o discard,defaults $PD_DEVICE $PD_MOUNT_POINT
  else
    break
  fi
done

# timezone
sudo timedatectl set-timezone America/Los_Angeles

touch /home/xinleic/.VM_READY