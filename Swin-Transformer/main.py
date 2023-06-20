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


class recorder:
    def __init__(self, writer, split='train'):
        # self.writer = writer
        global globe_writer
        self.writer = globe_writer
        self.split = split
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        if THRESHOLD:
            self.TP_3 = 0
            self.TN_3 = 0
            self.FP_3 = 0
            self.FN_3 = 0

            self.TP_4 = 0
            self.TN_4 = 0
            self.FP_4 = 0
            self.FN_4 = 0

        self.loss_count = 0
        self.epoch_count = 0

        self.verbose = False

    def thresh_updata_states(self, outputs, labels):

        softmax_output = softmax(outputs)
        for thr in [0.4, 0.3]:
            result_tmp = softmax_output.clone()
            if thr < 0.5:
                result_tmp[:, 1] = torch.where(result_tmp[:, 1] >= thr, 1, 0)
            elif thr > 0.5:
                result_tmp[:, 0] = torch.where(result_tmp[:, 0] >= 1 - thr, 1, 0)
            predictions = result_tmp.argmax(dim=-1)

            TP = torch.where((predictions == labels.cuda()) & (predictions == 1), 1, 0).sum()
            TN = torch.where((predictions == labels.cuda()) & (predictions == 0), 1, 0).sum()
            FP = torch.where((predictions != labels.cuda()) & (predictions == 1), 1, 0).sum()
            FN = torch.where((predictions != labels.cuda()) & (predictions == 0), 1, 0).sum()

            if thr == 0.4:
                self.TP_4 += TP
                self.TN_4 += TN
                self.FP_4 += FP
                self.FN_4 += FN
            elif thr == 0.3:
                self.TP_3 += TP
                self.TN_3 += TN
                self.FP_3 += FP
                self.FN_3 += FN

    def update_states(self, outputs, labels):
        if len(labels.shape) >= 2:
            labels = labels.argmax(dim=-1)

        predictions = outputs.argmax(dim=-1)

        self.TP += torch.where((predictions == labels.cuda()) & (predictions == 1), 1, 0).sum()
        self.TN += torch.where((predictions == labels.cuda()) & (predictions == 0), 1, 0).sum()
        self.FP += torch.where((predictions != labels.cuda()) & (predictions == 1), 1, 0).sum()
        self.FN += torch.where((predictions != labels.cuda()) & (predictions == 0), 1, 0).sum()


    def thresh_cal_metrics(self):
        accuracy_4 = (self.TP_4 + self.TN_4) / (self.TP_4 + self.TN_4 + self.FP_4 + self.FN_4)
        recall_4 = self.TP_4 / (self.TP_4 + self.FN_4)
        precision_4 = self.TP_4 / (self.FP_4 + self.TP_4)
        f1_score_4 = 2 * (recall_4 * precision_4) / (recall_4 + precision_4)

        print('-' * 10)
        print('threshold 0.4: ')
        print('accuracy: ', accuracy_4)
        print('recall: ', recall_4)
        print('precision: ', precision_4)
        print('f1:', f1_score_4)

        accuracy_3 = (self.TP_3 + self.TN_3) / (self.TP_3 + self.TN_3 + self.FP_3 + self.FN_3)
        recall_3 = self.TP_3 / (self.TP_3 + self.FN_3)
        precision_3 = self.TP_3 / (self.FP_3 + self.TP_3)
        f1_score_3 = 2 * (recall_3 * precision_3) / (recall_3 + precision_3)

        print('-' * 10)
        print('threshold 0.3: ')
        print('accuracy: ', accuracy_3)
        print('recall: ', recall_3)
        print('precision: ', precision_3)
        print('f1:', f1_score_3)

    def cal_metrics(self):
        accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        recall = self.TP / (self.TP + self.FN)
        precision = self.TP / (self.FP + self.TP)
        f1_score = 2 * (recall * precision) / (recall + precision)

        self.writer.add_scalar("Accuracy/" + self.split, accuracy, self.epoch_count)
        self.writer.add_scalar("Recall/" + self.split, recall, self.epoch_count)
        self.writer.add_scalar("Precision/" + self.split, precision, self.epoch_count)
        self.writer.add_scalar("f1/" + self.split, f1_score, self.epoch_count)
        self.writer.add_scalars("Confusion/" + self.split, {'TP': self.TP,
                                                            'TN': self.TN,
                                                            'FP': self.FP,
                                                            'FN': self.FN},
                                self.epoch_count)
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        self.epoch_count += 1

        print('accuracy: ', accuracy)
        print('recall: ', recall)
        print('precision: ', precision)
        print('f1:', f1_score)
        if self.verbose:
            sys.exit()

    def write_loss(self, loss):
        # global globe_writer
        self.writer.add_scalar("loss/" + self.split, loss, self.loss_count)
        # globe_writer.add_scalar("loss/"+self.split, loss, self.loss_count)

        self.loss_count += 1


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


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    if DEBUG:
        writer = SummaryWriter('test_writer')
    else:
        writer = SummaryWriter('no_froze_runs')
    train_recorder = recorder(writer, split='train')
    val_recorder = recorder(writer, split='val')

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        print('softtarget')
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
        print('label_smoothing')
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print('cross_entropy')
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).cuda())

    max_accuracy = 0.0

    # if config.TRAIN.AUTO_RESUME:
    #     resume_file = auto_resume_helper(config.OUTPUT)
    #     if resume_file:
    #         if config.MODEL.RESUME:
    #             logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
    #         config.defrost()
    #         config.MODEL.RESUME = resume_file
    #         config.freeze()
    #         logger.info(f'auto resuming from {resume_file}')
    #     else:
    #         logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, val_recorder)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    # if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
    #     load_pretrained(config, model_without_ddp, logger)
    #     acc1, acc5, loss = validate(config, data_loader_val, model, val_recorder)
    #     logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    # train_size = len(data_loader_train)

    for p in model.named_parameters():
        if p[1].requires_grad:
            print(p[0])

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        train_recorder)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

        acc1, acc5, loss = validate(config, data_loader_val, model, val_recorder)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, recorder):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        # print('here!!!!!!')
        if DEBUG:
            if idx > 10:
                break
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # cross_entropy_targets = torch.argmax(targets, dim=-1)
        # print(targets.shape)
        # targets = torch.argmax(targets, dim=-1)
        # print(targets.shape)

        # print('train before target shape:', targets.shape)
        # print('train before target:', targets[0])
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        # print(outputs)
        # print(targets.shape)
        # print(outputs.shape)
        recorder.update_states(outputs, targets)

        if config.TRAIN.ACCUMULATION_STEPS > 1:

            # print('target:', targets.shape)
            # print('ce_target:', cross_entropy_targets.shape)
            loss = criterion(outputs, targets)
            # loss = criterion(outputs, cross_entropy_targets)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            recorder.write_loss(loss)
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            loss = criterion(outputs, targets)
            recorder.write_loss(loss)
            optimizer.zero_grad()
            if config.AMP_OPT_LEVEL != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(amp.master_params(optimizer))
            else:
                loss.backward()
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    recorder.cal_metrics()


@torch.no_grad()
def validate(config, data_loader, model, recorder):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    pos_counter = 0
    neg_counter = 0
    COUNT = False

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        if DEBUG:
            if idx > 10:
                break
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # if target.any():
        #     print(target)
        #     print(target.sum())
        # print(images[0][0])
        # sys.exit()

        if COUNT:
            pos_counter += target.sum()
            neg_counter += (len(target) - target.sum())
            continue

        # compute output
        output = model(images)
        # print('before target:', target[0])
        recorder.update_states(output, target)

        if THRESHOLD:
            recorder.thresh_updata_states(output, target)

        # measure accuracy and record loss
        loss = criterion(output, target)
        recorder.write_loss(loss)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        acc1 = accuracy(output, target, topk=(1,))

        acc1 = reduce_tensor(acc1)
        # acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        # acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    if COUNT:
        print('Number of positives:', pos_counter)
        print('Number of negatives:', neg_counter)
        sys.exit()
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    recorder.cal_metrics()

    if THRESHOLD:
        recorder.thresh_cal_metrics()
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


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
