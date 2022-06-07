gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
# pip install 'jax[tpu]==0.3.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip uninstall -y clu
# pip install --upgrade 'clu @ git+https://github.com/google/CommonLoopUtils#egg=clu'

# pip install flax==0.4.1

# sudo pip3 uninstall -y tensorflow keras keras-nightly keras-preprocessing tensorboard tb-nightly tf-nightly tf-estimator-nightly
# pip3 uninstall -y tensorflow keras keras-nightly keras-preprocessing tensorboard tb-nightly tf-nightly tf-estimator-nightly
# pip3 install tensorflow==2.7

# pip3 install tensorflow-probability
# pip3 install tensorflow_addons

# pip3 install torch==1.7.1
# pip3 install torchvision==0.8.2
# pip3 install timm==0.4.12

# pip install cached_property
# pip install tensorstore

# pip3 list | grep 'jax\|flax\|tensorflow\|clu\|tf'

# python3 -c 'import tensorflow as tf; print(\"tf done!\")'
# python3 -c 'import jax; print(jax.device_count()); print(\"jax done!\")'
# python3 -c 'import flax; print(\"flax done!\")'

sudo apt-get -y update
sudo apt-get -y install nfs-common
sudo mkdir -p /kmh_data
sudo mount 10.60.38.146:/kmh_data /kmh_data
sudo chmod go+rw /kmh_data
ls /kmh_data
"