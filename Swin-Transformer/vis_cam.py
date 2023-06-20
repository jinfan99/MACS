# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.functional import softmax

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

import sys

from config import get_config
from models import build_model
from data import build_loader
# from models.my_build import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sys import exit
from pytorch_grad_cam import GradCAM, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import cv2
import os.path as osp
# from recorder import recorder

# globe_writer.add_scalar("test" , 1, 0)

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

DEBUG = False

THRESHOLD = False

if DEBUG:
    globe_writer = SummaryWriter('test_writer')
else:
    globe_writer = SummaryWriter('ipcai_swin_tiny_ratio_exp_1_freeze_seed_42_writer')


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='ipcai_output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config
# (136.074, 74.512, 62.077), (69.016, 61.919, 56.320)

def main(config):
    # dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    IMAGE_PATH = "/home/jinfan/repos/Swin-Transformer/img_test/bak"
    OUTPUT_PATH = "/home/jinfan/repos/Swin-Transformer/img_test/output"
    img_names = ['AK', 'AL', 'AO', 'AS', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'P', 'U', 'X', 'Y']

    model = build_model(config)
    load_pretrained(config, model, logger)
    model.cuda()

    for idx, n in enumerate(img_names):
        img_path = osp.join(IMAGE_PATH, n+'.jpg')
        # print(img_path)
        try:
            rgb_img = cv2.imread(img_path, 1)#[:, :, ::-1]
        except:
            continue
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[136.074/255, 74.512/255, 62.077/255],
                                        std=[69.016/255, 61.919/255, 56.320/255])
        # print(input_tensor.shape)

        logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

        # print(model)

        target_layers = [model.layers[-1].blocks[-1].norm1]
        # input_tensor = torch.zeros((1, 3, 224, 224))
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True, reshape_transform=reshape_transform)

        # cam = methods[args.method](model=model,
        #                            target_layers=target_layers,
        #                            use_cuda=args.use_cuda,
        #                            reshape_transform=reshape_transform)

        targets = [ClassifierOutputTarget(1)]

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)

        # print(grayscale_cam.shape)
        cv2.imwrite(osp.join(OUTPUT_PATH, 'tmp_map_'+n+'.jpg'), visualization)
        # cv2.imwrite(osp.join(OUTPUT_PATH, 'tmp_map_' + n + '_gray.jpg'), grayscale_cam)

    print('Done!')
    exit(0)

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
