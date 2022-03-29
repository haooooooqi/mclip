REPO=https://$GIT_ID:@github.com/KaimingHe/flax_dev.git
BRANCH=main

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
git config --global credential.helper store
git clone -b $BRANCH $REPO

pip install 'jax[tpu]==0.3.4' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade clu

pip install --upgrade jax
pip install --upgrade flax

pip3 install torchvision --upgrade
pip3 install tensorflow-probability
pip3 install tensorflow_addons

pip3 list | grep 'jax\|flax\|tensorflow '

"