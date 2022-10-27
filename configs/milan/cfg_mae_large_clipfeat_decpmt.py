import ml_collections

import configs.vit as vit
import configs.cfg_common_mae as cfg_common_mae


def get_config():
    """Get the hyperparameter configuration to train on TPUs."""
    config = cfg_common_mae.get_config()

    # mae img config
    config.model.model_img.mask_ratio = 0.75
    config.model.model_img.norm_pix_loss = True

    config.model.model_img.update(vit.get_l16_config())
    config.model.model_img.hidden_size = 1024
    config.model.model_img.transformer.mlp_dim = config.model.model_img.hidden_size * 4
    config.model.model_img.transformer.dropout_rate = 0.0
    config.model.model_img.transformer.droppath_rate = 0.0
    config.model.model_img.transformer.num_heads = 16
    config.model.model_img.transformer.num_layers = 24
    config.model.model_img.transformer.rescale_init = 1.0

    config.model.model_img.decoder.hidden_size = 512
    config.model.model_img.decoder.transformer = ml_collections.ConfigDict()
    config.model.model_img.decoder.transformer.mlp_dim = config.model.model_img.decoder.hidden_size * 4
    config.model.model_img.decoder.transformer.num_heads = 16
    config.model.model_img.decoder.transformer.num_layers = 8
    config.model.model_img.decoder.transformer.attention_dropout_rate = 0.0
    config.model.model_img.decoder.transformer.dropout_rate = 0.0
    config.model.model_img.decoder.transformer.droppath_rate = 0.0


    # prompt decoder
    config.model.model_img.use_prompt_decoder = True


    config.model.model_img.decoder.cross_attention = False

    config.model.model_txt_enable = False

    config.model.use_feat_target = True
    config.model.model_img_feat = ml_collections.ConfigDict()

    config.model.model_img_feat.update(vit.get_l14_config())
    config.model.model_img_feat.transformer.dropout_rate = 0.0
    config.model.model_img_feat.transformer.droppath_rate = 0.1
    config.model.model_img_feat.transformer.use_encoder_prenorm = True
    config.model.model_img_feat.transformer.quick_gelu = True  # true for openai pre-train
    config.model.model_img_feat.transformer.ln_eps = 1e-5
    config.model.model_img_feat.transformer.use_encoder_proj = True
    config.model.model_img_feat.transformer.use_encoder_proj_bias = False
    config.model.model_img_feat.transformer.proj_dim = 768
    config.model.model_img_feat.encoder_norm = True

    config.model_img_feat_pretrain_dir = "/checkpoint/ronghanghu/clip_jax_checkpoint_conversion/open_clip_to_jax_with_proj/ViT-B-16"
    config.model_img_feat_pretrain_flax = "clip"

    # txt
    config.model.model_txt_feat = ml_collections.ConfigDict()
    config.model.model_txt_feat.name = 'ViT-H_14'
    config.model.model_txt_feat.proj_dim = 768
    config.model.model_txt_feat.width = 768
    config.model.model_txt_feat.heads = 12
    config.model.model_txt_feat.layers = 12


    return config
