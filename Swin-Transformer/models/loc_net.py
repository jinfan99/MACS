import torch
import torch.nn as nn
import numpy as np

# from .transunet import TransUNet
from .vit_seg_modeling import VisionTransformer
from . import vit_seg_configs as configs
from .swin_transformer import SwinTransformer


class LocNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swin = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                    num_classes=2,
                                    embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                    depths=config.MODEL.SWIN.DEPTHS,
                                    num_heads=config.MODEL.SWIN.NUM_HEADS,
                                    window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                    mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                    qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                    qk_scale=config.MODEL.SWIN.QK_SCALE,
                                    drop_rate=config.MODEL.DROP_RATE,
                                    drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                    ape=config.MODEL.SWIN.APE,
                                    patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        config_vit = configs.get_r50_b16_config()
        config_vit.n_classes = 2
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224//16), int(224//16))
        self.seg_net = VisionTransformer(config_vit, img_size=224, num_classes=2).cuda()
        self.seg_net.load_from(weights=np.load('/home/jinfan/repos/Swin-Transformer/pretrained/R50+ViT-B_16.npz'))

    def forward(self, img):
        seg_pred = self.seg_net(img)
        # concat_input = torch.cat((img, seg_pred), dim=1)

        # print(concat_input.shape)

        class_pred = self.swin(torch.cat((img, seg_pred), dim=1))

        return class_pred, seg_pred
