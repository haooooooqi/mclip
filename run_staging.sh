VM_NAME=kmh-tpuvm-v3-256-4
echo $VM_NAME

# ------------------------------------------------
# copy all files to staging
# ------------------------------------------------
now=`date '+%y%m%d%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c8`
export STAGEDIR=/kmh_data/staging/${now}-${salt}-code

echo 'Copying files...'
rsync -a . $STAGEDIR --exclude=tmp
echo 'Done copying.'

sudo chmod 777 $STAGEDIR

cd $STAGEDIR
echo 'Current dir: '`pwd`
# ------------------------------------------------

source run_remote.sh
