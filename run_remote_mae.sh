echo 'code dir: '$STAGEDIR

# seed=0
batch=8192
lr=1e-4
ep=400

mask=0.0
mask_txt=0.0

txtw=0.1

tau=0.1
seed=42

partitions=1

rescale=1.0


dataset='imagenet2012:5.*.*'
dataset='imagenet_v2'
# dataset='imagenet_sketch'  # use local folder
dataset='dtd'
dataset='cars196'
dataset='oxford_flowers102'
dataset='food101'
dataset='sun397'
dataset='caltech101'
dataset='caltech_birds2011'
dataset='kitti'
dataset='voc'
dataset='mnist'
dataset='cifar10'
dataset='cifar100'
dataset='stl10'
dataset='imagenet2012_real'
dataset='imagenet_a'


######################
# batch=32
# VM_NAME=hf-mamut-v3-8
# VM_NAME=mamut-v3-32-1
######################




REMOTE_ROOT_FOLDER="checkpoint"
rescale=1.0

HASH=$(echo $RANDOM | md5sum | head -c 12; echo;)
CONFIG=milan/cfg_mae_base_clipfeat_decpmt
CONFIG=milan/cfg_mae_large_clipfeat_decpmt
# CONFIG=cfg_mae_base
# CONFIG=cfg_mae_large
# CONFIG=cfg_mae_large_fairclip

# _normpix_exwd_NOsplit_fastsave
JOBNAME=flax/${HASH}_$(date +%Y%m%d_%H%M%S)_maet5x_${VM_NAME}_${CONFIG}_${ep}ep_b${batch}_lr${lr}_mk${mask}txt${mask_txt}_s${seed}_p${partitions}st_re${rescale}_laion_a0.5_sanity_twoMAE_twoCross
RESUME=''

WORKDIR=gs://hf-gcp/checkpoints/${JOBNAME}
LOGDIR=/checkpoint/haoqifan/log/${JOBNAME}
REMOTE_LOGDIR=/${REMOTE_ROOT_FOLDER}/haoqifan/log/${JOBNAME}
mkdir -p ${LOGDIR}
chmod 777 ${LOGDIR}

PYTHONPATH_ORG=/checkpoint/haoqifan/workspace/clip_zero_shot
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
# export TFDS_DATA_DIR=/checkpoint/haoqifan/tensorflow_datasets
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
    --config.model.clr.tau=${tau} \
    --config.model.model_txt.decoder.cross_attention=True \
    --config.model.model_img.decoder.cross_attention=False \
    --config.model.model_txt.decoder.on_use=True \
    --config.model.model_img.decoder.on_use=True \
    --config.model.clr.clr_loss=False \
    --config.aug.txt.cls_token=False \
    --config.model.model_txt.decoder.loss_weight=${txtw} \
    --config.model.model_img.decoder.pool_x_part=False \
    --config.dataset=${dataset} \
    --config.model.clr.proj_layers=1 \
    --config.model.clr.proj_dim_out=512 \
    --config.model.model_proj.proj_layers=1 \
    --config.model.model_proj.proj_dim_out=512 \
    --config.aug.crop_ver=vc_png_pad \
    --config.model.model_img.patches.size=\(14\,14\) \
    --config.model_img_feat_pretrain_flax=clip_text \
    --config.model_img_feat_pretrain_dir=/checkpoint/haoqifan/clip_jax_checkpoint_conversion/open_clip_to_jax_with_txt_encoder/ViT-L-14-336px \
    --config.aug.txt.max_len=77 \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.tokenizer=hf_clip \
    --config.model.clip_pretrain=True \
    --config.model.model_img_feat.transformer.quick_gelu=False \
    --config.image_size 336 \
2>&1 | tee -a $REMOTE_LOGDIR/finetune_\$SSH_ID.log
" 2>&1 | tee -a $LOGDIR/finetune.log

    --config.dataset_path=/checkpoint/haoqifan/coco_caption_test/coco_cap_test.csv \
    --config.dataset_path=/checkpoint/haoqifan/datasets/flickr_test.csv \


    # config for Open CLIP ViT-L

    --config.model.model_img.patches.size=\(14\,14\) \
    --config.model_img_feat_pretrain_flax=clip_text \
    --config.model_img_feat_pretrain_dir=/checkpoint/haoqifan/clip_jax_checkpoint_conversion/open_clip_to_jax_with_txt_encoder/vit_l_14-laion400m_e32-3d133497 \
    --config.aug.txt.max_len=77 \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.tokenizer=hf_clip \
    --config.model.clip_pretrain=True \
    --config.model.model_img_feat.transformer.quick_gelu=False \



    # config for Open CLIP ViT-B
    --config.model.model_img.patches.size=\(14\,14\) \
    --config.model_img_feat_pretrain_flax=clip_text \
    --config.model_img_feat_pretrain_dir=/checkpoint/haoqifan/clip_jax_checkpoint_conversion/open_clip_to_jax_with_txt_encoder/vit_b_16-laion400m_e32-55e67d44 \
    --config.aug.txt.max_len=77 \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.tokenizer=hf_clip \
    --config.model.clip_pretrain=True \
    --config.model.model_img_feat.transformer.quick_gelu=False \




    # config for CLIP ViT-L 336
    --config.model.model_img.patches.size=\(14\,14\) \
    --config.model_img_feat_pretrain_flax=clip_text \
    --config.model_img_feat_pretrain_dir=/checkpoint/haoqifan/clip_jax_checkpoint_conversion/open_clip_to_jax_with_txt_encoder/ViT-L-14-336px \
    --config.aug.txt.max_len=77 \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.tokenizer=hf_clip \
    --config.model.clip_pretrain=True \
    --config.image_size 336 \


    # config for CLIP ViT-L
    --config.model.model_img.patches.size=\(14\,14\) \
    --config.model_img_feat_pretrain_flax=clip_text \
    --config.model_img_feat_pretrain_dir=/checkpoint/haoqifan/clip_jax_checkpoint_conversion/open_clip_to_jax_with_txt_encoder/ViT-L-14 \
    --config.aug.txt.max_len=77 \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.tokenizer=hf_clip \
    --config.model.clip_pretrain=True \

    # config for CLIP ViT-B
    --config.model_img_feat_pretrain_flax=clip_text \
    --config.model_img_feat_pretrain_dir=/checkpoint/haoqifan/clip_jax_checkpoint_conversion/open_clip_to_jax_with_txt_encoder/ViT-B-16 \
    --config.aug.txt.max_len=77 \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.tokenizer=hf_clip \
    --config.model.clip_pretrain=True \


    # config for ViT-L high resolution
    --config.model_img_feat_pretrain_dir=gs://kmh-gcp/checkpoints/flax/20220910_212550_maet5x_kmh-tpuvm-v3-512-1_cfg_mae_large_10000ep_b16384_lr4e-6_mk0.0txt0.0_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp/checkpoint_780000/ \
    --config.model_img_feat_pretrain_flax=fairclip_text_interp \
    --config.image_size 320 \

    # config for ViT-B
    --config.model_img_feat_pretrain_flax=fairclip_text \
    --config.model_img_feat_pretrain_dir=gs://kmh-gcp/checkpoints/flax/20220920_070424_maet5x_kmh-tpuvm-v3-256-4_cfg_mae_base_10000ep_b32768_lr4e-6_mk0.0txtNO_s100_p1st_re1.0_laion_a0.5_clrtau_ev7_512d1mlp_wd0.2_b0.98/checkpoint_390000/ \

    # config for ViT-L
    --config.model_img_feat_pretrain_flax=fairclip_text \
    --config.model_img_feat_pretrain_dir=gs://kmh-gcp/checkpoints/flax/20220910_212550_maet5x_kmh-tpuvm-v3-512-1_cfg_mae_large_10000ep_b16384_lr4e-6_mk0.0txt0.0_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp/checkpoint_780000/ \


    --config.model.model_txt.use_attention_mask=True \
    --config.model.model_txt.vocab_size=49408 \
    --config.aug.txt.max_len=77 \
    --config.aug.txt.tokenizer=hf_clip \



    --config.image_size=384 \
    --config.model_img_feat_pretrain_dir=gs://kmh-gcp/checkpoints/flax/20220914_202349_maet5x_kmh-tpuvm-v3-256-3_cfg_mae_base_10000ep_b32768_lr4e-6_mk0.0txtNO_s100_p1st_re1.0_laion_a0.5_clrtau_eval_512d1mlp_hfclip77b_autoreg_wd0.2_b0.98/checkpoint_99450/ \
    --config.model_img_feat_pretrain_dir=gs://kmh-gcp/checkpoints/flax/20220920_064511_maet5x_kmh-tpuvm-v3-512-1_cfg_mae_large_10000ep_b16384_lr4e-6_mk0.0txtNO_s100_p1st_re1.0_laion_a0.5_clrtau_ev7_512d1mlp_hfclip77b_autoreg_wd0.2_b0.98/checkpoint_780000 \

echo ${VM_NAME}

    --config.aug.txt.batch_process=True \
