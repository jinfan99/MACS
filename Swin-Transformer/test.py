import torch
import argparse
import numpy as np
import random
from models import build_model
from utils import recorder
from tqdm import tqdm
from config import get_config
import os
import torch.backends.cudnn as cudnn
from data import build_loader
import torch.distributed as dist
from logger import create_logger
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter


MODEL_NAME = 'swin_tiny'
EXP_INDEX = '0'
FREEZE_MODEL = False
CKPT_PATH = "/home/jinfan/repos/Swin-Transformer/ipcai_output/ipcai_swin_tiny_ratio_exp_all_freeze/default/ckpt_epoch_299.pth"

DEBUG = False

def main(config):
    _, dataset_val, _, data_loader_val, mixup_fn = build_loader(config)
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    load_pretrained(model_without_ddp)

    # start testing
    test(config, data_loader_val, model_without_ddp)


def test(config, data_loader_val, model):
    model.eval()

    test_recorder = recorder(writer_name='tmp_writer', split='test')

    all_pos = 0
    all_neg = 0

    for idx, (images, target) in tqdm(enumerate(data_loader_val)):
        if DEBUG:
            if idx > 10:
                break
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        tot = len(target)
        pos = torch.sum(target)
        neg = tot - pos
        all_pos += pos
        all_neg += neg


        output = model(images)
        softmax_output = softmax(output)[:15]
        gt = torch.tensor([1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])


        for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print('-'*10)
            print('threshold: ', str(thr))
            result_tmp = softmax_output.clone()
            if thr < 0.5:
                result_tmp[:, 1] = torch.where(result_tmp[:, 1] >= thr, 1, 0)
            elif thr > 0.5:
                result_tmp[:, 0] = torch.where(result_tmp[:, 0] >= 1 - thr, 1, 0)

            predictions = result_tmp.argmax(dim=-1)
            print(predictions[:15])

            TP = torch.where((predictions == gt.cuda()) & (predictions == 1), 1, 0).sum()
            TN = torch.where((predictions == gt.cuda()) & (predictions == 0), 1, 0).sum()
            FP = torch.where((predictions != gt.cuda()) & (predictions == 1), 1, 0).sum()
            FN = torch.where((predictions != gt.cuda()) & (predictions == 0), 1, 0).sum()

            accuracy = (TP + TN) / (TP + TN + FP + FN)
            recall = TP / (TP + FN)
            precision = TP / (FP + TP)
            f1_score = 2 * (recall * precision) / (recall + precision)
            print('accuracy: ', accuracy.data)
            print('recall: ', recall.data)
            print('precision: ', precision.data)
            print('f1_score: ', f1_score.data)
            print('-' * 10)

        # print(softmax(output))
        # print(softmax(output))
        # print('\n', len(predictions), '\n')

        # print(predictions[:15])
        # test_recorder.update_states(softmax(output), target, threshold=0.99)

    # test_recorder.cal_metrics()
    # print('acc:', test_recorder.accuracy)
    # print('recall:', test_recorder.recall)
    # print('precision:', test_recorder.precision)
    # print('f1:', test_recorder.f1_score)
    # print('TP:', test_recorder.TP)
    # print('TN:', test_recorder.TN)
    # print('FP:', test_recorder.FP)
    # print('FN:', test_recorder.FN)
    #
    # print('tot neg: ', all_neg)
    # print('tot pos: ', all_pos)


def load_pretrained(model, ckpt_path=CKPT_PATH):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint


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
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", default='3', type=int, required=False, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

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

    seed = 1
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
    # logger.info(config.dump())

    main(config)


