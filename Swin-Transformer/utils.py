# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import sys


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    # logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # # delete relative_position_index since we always re-init it
    # relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    # for k in relative_position_index_keys:
    #     del state_dict[k]
    #
    # # delete relative_coords_table since we always re-init it
    # relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    # for k in relative_position_index_keys:
    #     del state_dict[k]
    #
    # # delete attn_mask since we always re-init it
    # attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    # for k in attn_mask_keys:
    #     del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)

    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    # rt = tensor.clone()
    # print(tensor)
    rt = torch.tensor(tensor).cuda()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

# def _freeze_layer(model, freeze_stage):

class recorder:
    def __init__(self, writer_name='tmp_writer', split='train'):
        self.writer = SummaryWriter(writer_name)
        self.split = split
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.loss_count = 0
        self.epoch_count = 0

        self.verbose = False

    def update_states(self, outputs, labels, threshold=0.5):
        # print(self.split)
        # # print(outputs)
        # # print(labels)
        # print(outputs.shape)
        # print(labels.shape)
        # # print(labels[0])
        # print('--------')
        # print(outputs.shape)
        # sys.exit()
        outputs[:, 1] = torch.where(outputs[:, 1] >= threshold, 1, 0)         # threshold for 1
        # outputs[:, 0] = torch.where(outputs[:, 0] >= threshold, 1, 0)
        predictions = outputs.argmax(dim=-1)
        if len(labels.shape) >= 2:
            labels = labels.argmax(dim=-1)

        # print(predictions.shape)
        # print(labels.shape)
        # print('-------')
        self.TP += torch.where((predictions == labels.cuda()) & (predictions == 1), 1, 0).sum()
        self.TN += torch.where((predictions == labels.cuda()) & (predictions == 0), 1, 0).sum()
        self.FP += torch.where((predictions != labels.cuda()) & (predictions == 1), 1, 0).sum()
        self.FN += torch.where((predictions != labels.cuda()) & (predictions == 0), 1, 0).sum()

        if self.verbose and self.split == 'val':
            # if labels.any():
                print(labels)
            # if self.TP != 0 or self.FN != 0:
            #     print(self.loss_count, self.TP, 'TP')
            #     print(self.loss_count, self.FN, 'FN')

    def cal_metrics(self):
        self.accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.recall = self.TP / (self.TP + self.FN)
        self.precision = self.TP / (self.FP + self.TP)
        self.f1_score = 2 * (self.recall * self.precision) / (self.recall + self.precision)

        # self.writer.add_scalar("Accuracy/" + self.split, accuracy, self.epoch_count)
        # self.writer.add_scalar("Recall/" + self.split, recall, self.epoch_count)
        # self.writer.add_scalar("Precision/" + self.split, precision, self.epoch_count)
        # self.writer.add_scalar("f1/" + self.split, f1_score, self.epoch_count)
        # self.writer.add_scalars("Confusion/" + self.split, {'TP': self.TP,
        #                                                     'TN': self.TN,
        #                                                     'FP': self.FP,
        #                                                     'FN': self.FN},
        #                         self.epoch_count)
        # self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        # self.epoch_count += 1
        #
        # if self.verbose:
        #     sys.exit()

    def write_loss(self, loss):
        # global globe_writer
        self.writer.add_scalar("loss/" + self.split, loss, self.loss_count)
        # globe_writer.add_scalar("loss/"+self.split, loss, self.loss_count)

        self.loss_count += 1


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)

    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = torch.sum(pred * target)
    pred_sum = torch.sum(pred * pred)
    target_sum = torch.sum(target * target)

    return 1 - ((2. * intersection + 1e-5) / (pred_sum + target_sum + 1e-5))