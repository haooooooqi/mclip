echo 'code dir: '$STAGEDIR

# seed=0
batch=32768  # 4096, 8192, 16384, 32768
lr=4e-6  # MAE base lr: 1e-4; CLIP base lr: 5e-4/32768*256=3.90625e-06
ep=10000  # 10000  # 400M * 30 / 1.28M = 9375; 400M * 32 / 1.28M = 9375

mask=0.0
mask_txt=0.0

txtw=0

tau=0.01
seed=42

partitions=1

mask=0.75
mask=0.5
ep=2000
batch=24576  # 16384 + 8192
d_depth=4


######################
# batch=64
# VM_NAME=hf-mamut-v3-8
######################



REMOTE_ROOT_FOLDER="checkpoint"
rescale=1.0

HASH=$(echo $RANDOM | md5sum | head -c 8; echo;)
vitsize=base
CONFIG=cfg_mae_${vitsize}

# _normpix_exwd_NOsplit_fastsave
JOBNAME=flax/${HASH}_$(date +%Y%m%d_%H%M%S)_maet5x_${VM_NAME}_${CONFIG}_${ep}ep_b${batch}_lr${lr}_mk${mask}txt${mask_txt}_s${seed}_p${partitions}st_re${rescale}_laion_a0.5_sanity_twoMAE_twoCross
RESUME=''
# RESUME='gs://kmh-gcp/checkpoints/flax/20220910_212550_maet5x_kmh-tpuvm-v3-512-1_cfg_mae_large_10000ep_b16384_lr4e-6_mk0.0txt0.0_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp'

WORKDIR=gs://hf-gcp/checkpoints/${JOBNAME}
LOGDIR=/checkpoint/haoqifan/log/${JOBNAME}
REMOTE_LOGDIR=/${REMOTE_ROOT_FOLDER}/haoqifan/log/${JOBNAME}
mkdir -p ${LOGDIR}
chmod 777 ${LOGDIR}

PYTHONPATH_ORG=/checkpoint/haoqifan/workspace/mclip
PYTHONPATH_TAR=/checkpoint/haoqifan/jobs/${HASH}

REMOTE_PYTHONPATH_TAR=/${REMOTE_ROOT_FOLDER}/haoqifan/jobs/${HASH}
cp -R $PYTHONPATH_ORG $PYTHONPATH_TAR

echo $HASH
echo $PYTHONPATH_TAR


# source run_init_remote.sh

cd ${HOME} && gcloud alpha compute tpus tpu-vm ssh ${VM_NAME} --zone europe-west4-a \
    --worker=all --command "
# cd $STAGEDIR
# git config --global --add safe.directory $STAGEDIR

echo Current commit: $(git show -s --format=%h)
echo Current dir: $(pwd)

export GOOGLE_APPLICATION_CREDENTIALS=~/gcp_credential.json
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=8589934592
export TFDS_DATA_DIR=gs://kmh-gcp/tensorflow_datasets

echo REMOTE_PYTHONPATH_TAR $REMOTE_PYTHONPATH_TAR
cd ${REMOTE_PYTHONPATH_TAR}
echo Current dir: $(pwd)
source run_get_ssh_id.sh
python3 main.py \
    --workdir=${WORKDIR} \
    --config=configs/$CONFIG.py \
    --config.batch_size=${batch} \
    --config.log_every_steps=100 \
    --config.num_epochs=${ep} \
    --config.learning_rate=${lr} \
    --config.profile_memory=True \
    --config.model.model_img.transformer.rescale_init=${rescale} \
    --config.model.model_img.norm_pix_loss=True \
    --config.model.model_img.sincos=True \
    --config.model.model_img.mask_ratio=${mask} \
    --config.model.model_txt.mask_ratio=${mask_txt} \
    --config.seed_tf=${seed} \
    --config.seed_jax=${seed} \
    --config.seed_pt=${seed} \
    --config.partitioning.num_partitions=${partitions} \
    --config.opt_type=adamw \
    --config.opt_mu_dtype=float32 \
    --config.partitioning.partition_states=False \
    --config.resume_dir=${RESUME} \
    --config.aug.area_range=\(0.5\,1.0\) \
    --config.aug.flip=True \
    --config.model.clr.tau=${tau} \
    --config.model.model_txt.decoder.cross_attention=False \
    --config.model.model_img.decoder.cross_attention=False \
    --config.model.model_txt.decoder.on_use=False \
    --config.model.model_img.decoder.on_use=False \
    --config.model.clr.clr_loss=True \
    --config.aug.txt.cls_token=False \
    --config.model.model_txt.decoder.loss_weight=${txtw} \
    --config.model.clr.proj_layers=1 \
    --config.model.clr.proj_dim_out=512 \
    --config.model.model_proj.proj_layers=1 \
    --config.model.model_proj.proj_dim_out=512 \
    --config.model.model_img.decoder.hidden_size=768 \
    --config.model.model_img.decoder.transformer.num_heads=12 \
    --config.model.clr.tau_learnable=True \
    --config.opt.b2=0.98 \
    --config.opt.weight_decay=0.2 \
    --config.eval_only=False \
    --config.aug.eval_pad=0 \
    --config.model.model_img.decoder.transformer.num_layers=${d_depth} \
    --config.model.clr.contrast_with_mask_only=False \
    --config.model.model_img.decoder.prompt_attention=False \
    --config.model.model_img.decoder.no_attention=True \
    --config.model.clr.bp2txt=True \
    --config.model.clr.mean_loss=False \
2>&1 | tee -a $REMOTE_LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee -a $LOGDIR/finetune.log

    # --config.aug.txt.tokenizer=hf_clip \
    # --config.aug.txt.max_len=77 \
    # --config.model.model_txt.vocab_size=49408 \
    # --config.aug.txt.batch_process=True \

    # --config.opt.b2=0.98 \
    # --config.opt.weight_decay=0.2 \

    # --config.aug.txt.tokenizer=hf_clip \
    # --config.aug.txt.max_len=77 \
    # --config.model.model_txt.vocab_size=49408 \
    # --config.aug.txt.batch_process=True \
    # --config.model.model_txt.use_attention_mask=True \


echo ${VM_NAME}
