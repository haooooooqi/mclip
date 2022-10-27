import ml_collections

import configs.vit as vit


def get_config():
    """Get the hyperparameter configuration to train on TPUs."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 1e-3  # this is the base lr
    config.warmup_epochs = 5.0
    config.min_abs_lr = 1e-6  # this is abs lr
    config.warmup_abs_lr = 1e-6  # this is abs lr
    config.learning_rate_decay = 0.75  # lrd

    config.num_epochs = 100.0
    config.log_every_steps = 100
    config.save_every_epochs = 10

    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    # Consider setting the batch size to max(tpu_chips * 256, 8 * 1024) if you
    # train on a larger pod slice.
    config.batch_size = 1024
    config.cache = True

    # model config
    config.model = vit.get_b16_config()  # ViT-B/16
    config.model.transformer.dropout_rate = 0.0
    config.model.transformer.droppath_rate = 0.1
    config.model.num_classes = 1000

    # optimizer config
    config.opt_type = "adamw"
    config.opt = ml_collections.ConfigDict()
    config.opt.b1 = 0.9
    config.opt.b2 = 0.999
    config.opt.weight_decay = 0.05
    config.opt_mu_dtype = "float32"

    config.exclude_wd = True  # exclude some weight decays (bias, norm, cls, posembed)

    # aug config
    config.aug = ml_collections.ConfigDict()
    config.aug.area_range = (0.08, 1)
    config.aug.aspect_ratio_range = (3.0 / 4, 4.0 / 3.0)
    config.aug.crop_ver = "v4"  # v1, v3
    config.aug.label_smoothing = 0.1
    config.aug.autoaug = "randaugv2"  # autoaug, randaug, randaugv2, or None
    config.aug.color_jit = None  # [0.4, 0.4, 0.4]  # None to disable; [brightness, contrast, saturation]

    # mixup config
    config.aug.mix = ml_collections.ConfigDict()
    config.aug.mix.mixup = True
    config.aug.mix.mixup_alpha = 0.8
    config.aug.mix.cutmix = True
    config.aug.mix.cutmix_alpha = 1.0
    config.aug.mix.mode = "batch"  # batch, pair, elem: for timm mixup only

    # rand erase config
    config.aug.randerase = ml_collections.ConfigDict()
    config.aug.randerase.on = True
    config.aug.randerase.prob = 0.25

    # init config
    config.rescale_init = False  # rescale initialized weights by layer id
    config.model.rescale_head_init = 0.001  # rescale the head initialized weights

    # memory and profiling
    config.profile_memory = False
    config.profile = ml_collections.ConfigDict()
    config.profile.use_profile_server = False
    config.profile.profile_server_port = 9999
    config.profile.profile_start_step = 50
    config.profile.profile_num_steps = 10

    # utils
    config.resume_dir = ""

    config.pretrain_dir = ""
    config.pretrain_fmt = "t5x"  # 'jax' or 't5x'

    config.eval_only = False

    # seeds
    config.seed_jax = 0
    config.seed_tf = 0
    config.seed_pt = 0

    # torchload
    config.torchload = ml_collections.ConfigDict()
    config.torchload.data_dir = "/kmh_data/imagenet_full_size/061417"
    config.torchload.num_workers = 32

    # partitioning
    config.partitioning = ml_collections.ConfigDict()
    config.partitioning.num_partitions = 1
    config.partitioning.partition_states = False
    #config.partitioning.force_partition_states_data_first = False
    #config.partitioning.partition_states_for_encoder_only = False
    # config.partitioning.activation_partitioning_dims = 1
    #config.partitioning.parameter_partitioning_dims = 1

    return config
