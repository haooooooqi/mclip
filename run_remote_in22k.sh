echo 'code dir: '$STAGEDIR

# seed=0
batch=1024
lr=1e-3
wd=0.05
lrd=0.75
ep=50
warm=5
dp=0.2
pdp=0.0
beta2=0.999

partitions=1

pft=0  # predictor layers for ft
stopg=0  # number of stopgrad blocks

vitsize=huge1x_p16 # large
CONFIG=cfg_vit_${vitsize}

source scripts/select_chkpt_${vitsize}.sh
name=`basename ${PRETRAIN_DIR}`
# name=`basename $(dirname ${PRETRAIN_DIR})`


# finetune_pytorch_recipe (ftpy): lb0.1_b0.999_cropv4_exwd_initv2_headinit0.001_tgap_dp_mixup32_cutmix32_noerase_warmlr_minlr_autoaug
# finetune_torch_loader (fttl): randaugv2erase_TorchLoader
JOBNAME=flax/${name}_finetune/$(date +%Y%m%d_%H%M%S)_${VM_NAME}_${CONFIG}_${ep}ep_fttl_IN22K_b${batch}_wd${wd}_lr${lr}_lrd${lrd}_pdp${pdp}_dp${dp}_warm${warm}_s${seed}_beta${beta2}_p${partitions}st_stop${stopg}_pred${pft}_act1_param1
RESUME=''
# RESUME='gs://kmh-gcp/checkpoints/flax/20220606_004214_maet5x_kmh-tpuvm-v3-512-2_cfg_mae_huge4x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p8_re0.5_normpix_exwd_split_fastsave_finetune/20220629_164557_kmh-tpuvm-v3-512-2_cfg_vit_huge4x_p16_50ep_fttl_IN22K_b1024_wd0.05_lr1e-3_lrd0.75_pdp0.0_dp0.2_warm5_s0_beta0.999_p8st_stop16_helloworld_resume2'
# RESUME='gs://kmh-gcp/checkpoints/flax/20220606_004214_maet5x_kmh-tpuvm-v3-512-2_cfg_mae_huge4x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p8_re0.5_normpix_exwd_split_fastsave_finetune/20220701_024255_kmh-tpuvm-v3-512-2_cfg_vit_huge4x_p16_50ep_fttl_IN22K_b1024_wd0.05_lr1e-3_lrd0.75_pdp0.0_dp0.2_warm5_s0_beta0.999_p8st_stop16_saveDBG'
# RESUME='gs://kmh-gcp/checkpoints/flax/20220606_004214_maet5x_kmh-tpuvm-v3-512-2_cfg_mae_huge4x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p8_re0.5_normpix_exwd_split_fastsave_finetune/20220701_184553_kmh-tpuvm-v3-256-7_cfg_vit_huge4x_p16_50ep_fttl_IN22K_b1024_wd0.05_lr1e-3_lrd0.75_pdp0.0_dp0.2_warm5_s0_beta0.999_p8st_stop16_reduce'
# RESUME='gs://kmh-gcp/checkpoints/flax/20220606_070525_maet5x_kmh-tpuvm-v3-256-3_cfg_mae_huge3x_p16_1600ep_b4096_lr1e-4_mk0.75_s100_p4_re1.0_normpix_exwd_split_fastsave_finetune/20220705_051632_kmh-tpuvm-v3-256-6_cfg_vit_huge3x_p16_50ep_fttl_IN22K_b1024_wd0.05_lr1e-3_lrd0.75_pdp0.0_dp0.2_warm5_s0_beta0.999_p8st_stop16_saveDBG2'

WORKDIR=gs://kmh-gcp/checkpoints/${JOBNAME}
LOGDIR=/kmh_data/logs/${JOBNAME}
mkdir -p ${LOGDIR}
chmod 777 ${LOGDIR}

# source run_init_remote.sh

gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
cd $STAGEDIR
git config --global --add safe.directory $STAGEDIR

echo Current commit: $(git show -s --format=%h)
echo Current dir: $(pwd)

export GOOGLE_APPLICATION_CREDENTIALS=~/gcp_credential.json
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets
export GOOGLE_APPLICATION_CREDENTIALS=~/gcp_credential.json

source run_get_ssh_id.sh

python3 main.py \
    --workdir=${WORKDIR} \
    --config=configs/$CONFIG.py \
    --config.pretrain_dir=${PRETRAIN_DIR} \
    --config.batch_size=${batch} \
    --config.learning_rate=${lr} \
    --config.learning_rate_decay=${lrd} \
    --config.opt.weight_decay=${wd} \
    --config.opt.b2=${beta2} \
    --config.warmup_epochs=${warm} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.save_every_epochs=1 \
    --config.profile_memory=True \
    --config.donate=True \
    --config.init_backend=tpu \
    --config.aug.mix.mixup=True \
    --config.aug.mix.cutmix=True \
    --config.aug.randerase.on=True \
    --config.aug.autoaug=randaugv2 \
    --config.model.transformer.droppath_rate=${dp} \
    --config.model.predictor.transformer.droppath_rate=${pdp} \
    --config.model.predictor.transformer.num_layers=${pft} \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.model.classifier=token \
    --config.partitioning.num_partitions=${partitions} \
    --config.pretrain_fmt=t5x \
    --config.model.sincos=True \
    --config.model.adapter.on_use=False \
    --config.model.stopgrad_blocks=${stopg} \
    --config.partitioning.partition_states=False \
    --config.partitioning.activation_partitioning_dims=1 \
    --config.partitioning.parameter_partitioning_dims=1 \
    --config.resume_dir=$RESUME \
    --config.torchload.data_dir=/datasets/imagenet-22k \
    --config.model.num_classes=21841 \
2>&1 | tee -a $LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee -a $LOGDIR/finetune.log

echo ${VM_NAME}