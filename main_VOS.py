# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter

import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from augs import DataAugmentationDINO

import random
from mask_generator import RandomMaskingGenerator
from utils import _sample_index, _sample_from
from dataset import make_dataloader_vrw
from losses import DINOLoss, calc_cross_frame_dino, calc_mim_loss, calc_affine

import warnings
warnings.filterwarnings("ignore")

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)
    parser.add_argument('--dataset', default='', type=str,
        choices=['ytvos_all','charades', 'kinetics400'] ,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")

    # Arch parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] 
                + torchvision_archs + torch.hub.list("facebookresearch/xcit"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--img_size', default=64, type=int, help='Imgae size used for creating vit, determining the initialized positional embedding size.')
    parser.add_argument('--out_dim', default=4096, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--nlayers', default=3, type=int, help='num layers for DINOHead')
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")

    # augmentation params
    parser.add_argument('--ori_resize', type=float, nargs='+', default=(720,1280),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument('--at_downscale_factor', default=0.5, type=float,
        help="""The factor of original img size to be visualized in tensorboard.""")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_number', type=int, default=1, help="""Number of global crops.""")
    parser.add_argument('--global_crops_size', default=224, type=int, help="""Output global crop size after random cropping.""")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.80, 0.95), help="""Random crop scale range for global crops.""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of local crops.""")
    parser.add_argument('--local_crops_size', type=int, default=64, help="""Output local crop size after random cropping.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.80), help="""Random crop scale range for local crops.""")

    # PATHs
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str, help='Please specify path to the ImageNet training data.')
    parser.add_argument('--csv_path', default='./ytvos.csv')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")

    # running parameters
    parser.add_argument('--distributed', action='store_true', help='Whether to use distrited parallel')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=24, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--start_epoch", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--lr-milestones', nargs='+', default=[20, 30, 40], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.3, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')

    # dataloader parameters
    parser.add_argument('--cache_path', default='', type=str, help="suffix of save name")
    parser.add_argument("--frame_skip", default=8, type=int, help='Frame skip for dataset sampler. ')
    parser.add_argument("--seq_length", default=4, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--clips_per_video", default=5, type=int, help='...')

    # Loss cfdino = out_g2g + out_l2g
    parser.add_argument('--weight_cfdino', type=float, default=1.0, help='Cfdino loss weight. Cfdino = out_g2g + out_l2g.')

    # Loss in_mim 
    parser.add_argument('--weight_mim', type=float, default=1.0, help='in_mim loss weight.')
    parser.add_argument('--mask_ratio', nargs='+', type=float, default=(0.1,0.5),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")
    parser.add_argument("--mask_prob", default=0.5, type=float, help='the probility of using MIM')

    # Loss in_aff 
    parser.add_argument('--weight_affine', type=float, default=1.0, help='in_affinity loss weight.' )
    parser.add_argument("--affine_teacher_temp_s", default=0.04, type=float, help='...')
    parser.add_argument("--affine_teacher_temp_e", default=0.04, type=float, help='...')
    parser.add_argument("--affine_student_temp", default=0.1, type=float, help='...')
    
    # MISCs
    parser.add_argument("--multi_scale_layer", default=1, type=int, nargs='+', 
                        help="Last n layers for multi scale trianning. Please ignore it. We only use the last layer for the final version")
    parser.add_argument('--remark', default='', type=str, help="suffix of save name")

    return parser

def make_schedulers(args, data_loader):

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * args.seq_length * utils.get_world_size()) / (256*4.),  # linear scaling rule # 2*8*4/256*4
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                            args.epochs, len(data_loader))
    weight_rw_schedule = utils.cosine_scheduler(1, 0, args.epochs, len(data_loader))
    # affine teacher temp scheduler 
    affine_teacher_temp_schedule = (
            np.linspace(args.affine_teacher_temp_s,
                        args.affine_teacher_temp_e, 
                        args.epochs)
            # np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        )

    return lr_schedule, wd_schedule, momentum_schedule, weight_rw_schedule, affine_teacher_temp_schedule

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss_cls_list, dino_loss_hw_list, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, weight_rw_schedule, affine_teacher_temp_schedule, epoch,
                    fp16_scaler, swriter, args, rg):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    affine_teacher_temp = affine_teacher_temp_schedule[epoch] # warmup affine teacher temp for each epoch

    for it, (images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        
        # ====== get mask for current iteration ======
        mask_frame = torch.from_numpy(np.stack([rg()] * args.batch_size_per_gpu)).bool().unsqueeze(0).unsqueeze(0).expand(args.seq_length, args.global_crops_number, -1, -1).cuda() # T, 1, B, N
        torch.distributed.broadcast(mask_frame, src=0) # broadcast to all devices to ensure that the mask token number is the same.

        # ================ interperate images ================
        global_frames = []
        local_frames = []
        images_at_frames = []
        ng = args.global_crops_number
        nl = args.local_crops_number
        # global x 2; grid_quart x 2; grid_quart_cj x 2; local x 8; g_params; l_params; img_at;
        for frames in images:
            global_frames.append(frames[:ng])
            local_frames.append(frames[ng:ng+nl])
            images_at_frames.append(frames[-1]) # bs, num_local_crops, 4

        # ================ visualize augmented crops to tensorboard ==================
        if utils.is_main_process() and it==0:
            for frame_idx, (ori, glb, local) in enumerate(zip(images_at_frames, global_frames, local_frames)):

                img_ori = ori[0].unsqueeze(0)
                img_ori = torchvision.utils.make_grid(img_ori, normalize=True, scale_each=True, nrow=img_ori.shape[0])

                global_views = torch.stack(glb)[:,0]
                global_views = torchvision.utils.make_grid(global_views, normalize=True, scale_each=True, nrow=global_views.shape[0])

                swriter.add_image('frame_{}_global_views'.format(frame_idx), global_views, global_step=epoch)
                swriter.add_image('frame_{}_img_ori'.format(frame_idx), img_ori, global_step=epoch)

                if local:
                    local_views = torch.stack(local)[:,0]
                    local_views = torchvision.utils.make_grid(local_views, normalize=True, scale_each=True, nrow=local_views.shape[0])  
                    swriter.add_image('frame_{}_local_views'.format(frame_idx), local_views, global_step=epoch)

        # ====== update weight decay and learning rate ======
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # ====== input to cuda ===== 
        def to_cuda(images_crops):
            return [ [im.cuda(non_blocking=True) for im in frames] for frames in images_crops]  
        global_frames = to_cuda(global_frames)
        local_frames =  to_cuda(local_frames)

        # ====== teacher and student forward + compute loss =======
        with torch.cuda.amp.autocast(fp16_scaler is not None):

            loss_dummy = torch.tensor(0.).cuda()
            glb_teacher_frame = []
            glb_student_frame = []
            local_student_frame = []

            # ===== batch forward ====== 
            global_frames = utils.list2tensor(global_frames) # T, NG, B, 3, 256, 256
            local_frames = utils.list2tensor(local_frames) # T, NL, B, 3, 256,256
            T, NG, B, _, _, _,  = global_frames.shape # T*NG*B, 
            if local_frames is not None:
                T, NL, B, _, _, _,  = local_frames.shape

            # ===== prepare output tokens ====== 
            # only unmasked global crops are sent to the teacher module 
            _, cls_aft_prj, _, hw_aft_prj, _ = teacher([global_frames.flatten(0,2)], mask=None, return_hw=True, return_atten=True)
            glb_teacher_cls, glb_teacher_hw = cls_aft_prj[0], hw_aft_prj[0]
            # masked global crops (prob=0.5) and local crops are sent to the student module
            assert args.local_crops_number > 0
            if args.weight_mim or args.weight_affine: # Mask is used only when mim loss is used. We only mask the glb crops.
                _, cls_aft_prj, _, hw_aft_prj, = student([global_frames.flatten(0,2), local_frames.flatten(0,2)], mask=[mask_frame.flatten(0,2), None], return_hw=True, return_atten=False)
            else:
                _, cls_aft_prj, _, hw_aft_prj, = student([global_frames.flatten(0,2), local_frames.flatten(0,2)], mask=[None, None], return_hw=True, return_atten=False)
            glb_student_cls, local_student_cls = cls_aft_prj[0], cls_aft_prj[1]
            glb_student_hw, local_student_hw = hw_aft_prj[0], hw_aft_prj[1]
        
            # ===== Out-generative learning =====
            # 1. cf dino loss = out_g2g + out_l2g
            loss_cfdino = torch.tensor(0.).cuda()
            if args.weight_cfdino:

                def calc_cfdino(glb_teacher_cls, glb_student_cls, local_student_cls, dino_loss_cls):
                    
                    glb_teacher_frame = glb_teacher_cls.view(T, ng*B, -1)
                    glb_student_frame = glb_student_cls.view(T, ng*B, -1)
                    if local_student_cls is not None:
                        local_student_frame = local_student_cls.view(T, nl*B, -1)
                    else:
                        local_student_frame = None
                    loss_cfdino = calc_cross_frame_dino(args, glb_teacher_frame, glb_student_frame, local_student_frame, dino_loss_cls, epoch)
                    
                    return loss_cfdino

                for i in range(len(args.multi_scale_layer)):
                    assert args.global_crops_number==1
                    loss_cfdino += calc_cfdino(glb_teacher_cls[i], glb_student_cls[i], local_student_cls[i], dino_loss_cls_list[i])
                loss_cfdino /= len(args.multi_scale_layer)

            # ===== In-generative Learning =====
            # 2. in_mim loss 
            loss_mim = torch.tensor(0.).cuda()
            if args.weight_mim:

                def calc_mim(glb_teacher_hw, glb_student_hw, dino_loss_hw):
                    _glb_teacher_hw = glb_teacher_hw.view(T, ng*B, *glb_teacher_hw.shape[-2:]) # T, ng*B, 784, 4096
                    _glb_student_hw = glb_student_hw.view(T, ng*B, *glb_teacher_hw.shape[-2:]) # T, ng*B, 784, 4096
                    loss_mim = calc_mim_loss(args, _glb_student_hw, _glb_teacher_hw, mask_frame, dino_loss_hw, epoch)
                    return loss_mim

                for i in range(len(args.multi_scale_layer)):
                    loss_mim += calc_mim(glb_teacher_hw[i], glb_student_hw[i], dino_loss_hw_list[i])
                loss_mim /= len(args.multi_scale_layer)
                
            # 3. in_aff loss
            loss_affine = torch.tensor(0.).cuda()
            if args.weight_affine and not mask_frame.sum()==0:  

                for i in range(len(args.multi_scale_layer)):
                    loss_affine += calc_affine(args, mask_frame, affine_teacher_temp, glb_student_hw[i], glb_teacher_hw[i])
                loss_affine /= len(args.multi_scale_layer)
            
            # summary of losses
            loss = 0.0 * loss_dummy + args.weight_cfdino * loss_cfdino + args.weight_mim * loss_mim + args.weight_affine * loss_affine
        
            # add losses to tensorboard 
            if utils.is_main_process():
                if args.weight_cfdino:
                    swriter.add_scalar('loss_cfdino', loss_cfdino.item(), global_step=it)
                if args.weight_mim:
                    swriter.add_scalar('loss_mim', loss_mim.item(), global_step=it)
                if args.weight_affine:
                    swriter.add_scalar('loss_affine', loss_affine.item(), global_step=it)
                swriter.add_scalar('loss', loss.item(), global_step=it)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # ===== student gradient update =====
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # ===== EMA update for the teacher weight =====
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # ===== logging =====
        torch.cuda.synchronize()  
        metric_logger.update(loss_cfdino=loss_cfdino.item())
        if args.weight_mim:
            metric_logger.update(loss_mim=loss_mim.item())
        if args.weight_affine:
            metric_logger.update(loss_affine=loss_affine.item())
            metric_logger.update(affine_teacher_temp=affine_teacher_temp)
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if utils.is_main_process():
            swriter.add_scalar('loss', loss.item(), global_step=it)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_INO(args):

    if args.distributed: 
        print('!!! Using distributed parallel ! ')
        utils.init_distributed_mode(args)

    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    # ============ preparing data ... ============
    data_loader = make_dataloader_vrw(args, epoch=0)
  
    # ============ building student and teacher networks ... ============
    student, teacher, teacher_without_ddp = vits.make_model(args)

    # ===== preparing loss ... =====
    # We use seperate DINOLoss (containing catched batch statics) for cls tokens and hw tokens (in line with iBoT). 
    dino_loss_cls_list = utils.get_repeat_module_list(args, DINOLoss, len(args.multi_scale_layer), 
                                                args, 
                                                args.out_dim,
                                                (args.local_crops_number + 2),  # total number of crops = 2 global crops + local_crops_number
                                                args.warmup_teacher_temp,
                                                args.teacher_temp,
                                                args.warmup_teacher_temp_epochs,
                                                args.epochs).cuda()
    dino_loss_hw_list = utils.get_repeat_module_list(args, DINOLoss, len(args.multi_scale_layer), 
                                                args, 
                                                args.out_dim,
                                                (args.local_crops_number + 2),  # total number of crops = 2 global crops + local_crops_number
                                                args.warmup_teacher_temp,
                                                args.teacher_temp,
                                                args.warmup_teacher_temp_epochs,
                                                args.epochs).cuda()

    # ===== preparing optimizer ... =====
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw": # default
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ===== init schedulers ... =====
    lr_schedule, wd_schedule, momentum_schedule, weight_rw_schedule, affine_teacher_temp_schedule = make_schedulers(args, data_loader)
    print(f"Loss, optimizer and schedulers ready.")

    # ===== optionally resume training ... =====
    to_restore = {"epoch": 1}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss_cls_list=dino_loss_cls_list,
        dino_loss_hw_list=dino_loss_hw_list
        # dino_loss=dino_loss,
        # dino_loss_hw=dino_loss_hw,
        # dino_loss_affine=dino_loss_affine
    )
    start_epoch = to_restore["epoch"]

    # ===== loop of epochs =====
    rg = RandomMaskingGenerator(int(args.global_crops_size/args.patch_size), args.mask_ratio, mask_prob=args.mask_prob)
    swriter = SummaryWriter(args.output_dir + '/tensorboard')
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs+1):
        
        # train one epoch
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss_cls_list, dino_loss_hw_list,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule, weight_rw_schedule, affine_teacher_temp_schedule, 
            epoch, fp16_scaler, swriter, args, rg)
        
        # writing ckpt and logs
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss_cls_list': dino_loss_cls_list.state_dict(),
            'dino_loss_hw_list': dino_loss_hw_list.state_dict()
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {'time':time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), **{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':

    parser = argparse.ArgumentParser('INO', parents=[get_args_parser()])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, '_'.join([ 'Da', args.dataset, 'Ar', args.arch, str(args.patch_size), 'Lr', '[', str(args.lr), str(args.warmup_epochs), str(args.epochs), str(args.min_lr), ']', 
                                                             'Bs', str(args.batch_size_per_gpu), 'Rmk', args.remark]))
    os.makedirs(args.output_dir, exist_ok=True)

    # use fp16
    args.use_fp16 = True
    if args.use_fp16: print('!!! fp16 is on')
    
    train_INO(args)
