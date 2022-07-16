# VM_NAME=kmh-tpuvm-v3-128-1
VM_NAME=cx_512a
echo $VM_NAME

# ------------------------------------------------
# copy all files to staging
# ------------------------------------------------
now=`date '+%y%m%d%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`
export STAGEDIR=/shoubhikdn_data/staging/${now}-${salt}-code

echo 'Copying files...'
# rsync -a . $STAGEDIR --exclude=tmp
sudo mkdir -p $STAGEDIR
sudo cp -r . $STAGEDIR
echo 'Done copying.'

chmod 777 $STAGEDIR

cd $STAGEDIR
echo 'Current dir: '`pwd`
# ------------------------------------------------

for seed in 0 # 1 2 3 
do
for ep in 100 # 50 30
do
for lrd in 0.7 0.8 0.6 0.9 0.95
do
for lr in 1e-4 5e-4 1e-3 2e-3
do
for dp in 0.2
do
source run_remote_sweep.sh $lr $lrd $ep $dp

echo sleep 1m
sleep 1m
source run_kill_remote.sh

echo sleep 1m
sleep 1m
done
done
done
done
done
