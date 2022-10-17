"""
Single-scale inference
Copyright (c) 2021 TU Darmstadt
Author: Nikita Araslanov <nikita.araslanov@tu-darmstadt.de>
License: Apache License 2.0
"""

import os
import sys
import numpy as np
import imageio
import time
import argparse

import torch.multiprocessing as mp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as Transform

from utils import Timer
from utils import check_dir
from palette_davis import palette as davis_palette

from torch.utils.data import DataLoader
from dataset.Inference_Dataset import DataSeg

from labelprop.common import LabelPropVOS_DINO

from PIL import Image
# deterministic inference
from torch.backends import cudnn

import vision_transformer as vits
from utils import check_before_run




cudnn.enabled = True
cudnn.benchmark = False
cudnn.deterministic = True

VERBOSE = True


def free_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return f


def mask2rgb(mask, palette):
    mask_rgb = palette(mask)
    mask_rgb = mask_rgb[:,:,:3]
    return mask_rgb

def mask_overlay(mask, image, palette):
    """Creates an overlayed mask visualisation"""
    mask_rgb = mask2rgb(mask, palette)
    return 0.3 * image + 0.7 * mask_rgb

class ResultWriter:
    
    def __init__(self, palette, out_path):

        self.palette = palette
        self.out_path = out_path
        self.verbose = VERBOSE

    def save(self, frames, masks_pred, masks_gt, flags, fn, seq_name):
        
        # vos is the mask output
        subdir_vos = os.path.join(self.out_path, "vos")
        check_dir(subdir_vos, seq_name)

        # vis is the blended output
        subdir_vis = os.path.join(self.out_path, "vis")
        check_dir(subdir_vis, seq_name)

        for frame_id, mask in enumerate(masks_pred.split(1, 0)):

            mask = mask[0].numpy().astype(np.uint8)
            filepath = os.path.join(subdir_vos, seq_name, "{}.png".format(fn[frame_id][0]))

            # saving only every 5th frame
            if flags[frame_id] != 0:
                # print(mask.shape)
                imageio.imwrite(filepath, mask)

                if self.verbose:
                    frame = frames[frame_id].numpy()
                    #mask_gt = masks_gt[frame_id].numpy().astype(np.uint8)
                    #masks = np.concatenate([mask, mask_gt], 1)
                    #frame = np.concatenate([frame, frame], 2)
                    frame = np.transpose(frame, [1,2,0])

                    overlay = mask_overlay(mask, frame, self.palette)
                    filepath = os.path.join(subdir_vis, seq_name, "{}.png".format(fn[frame_id][0]))
                    imageio.imwrite(filepath, (overlay * 255.).astype(np.uint8))


def convert_dict(state_dict):   
    new_dict = {}
    for k,v in state_dict.items():
        new_key = k.replace("module.", "")
        new_dict[new_key] = v
    return new_dict

def mask2tensor(mask, idx, num_classes=7):
    # h,w -> 1, C, h, w
    h,w = mask.shape
    mask_t = torch.zeros(1,num_classes,h,w)
    mask_t[0, idx] = mask

    return mask_t

def configure_tracks(args, masks_gt, tracks, num_objects):
    """Selecting masks for initialisation

    Args:
        masks_gt: [T,H,W]
        tracks: [T]

    """
    # NOTE: For DAVIS, masks are given only at the first frame, 
    # while for YTVOS, the masks will be gradually included from intermediate frames)
    
    init_masks = {} # {first time step: mask} 

    # we always have first mask
    # if there are no instances, it will be simply zero
    H,W = masks_gt[0].shape[-2:]
    init_masks[0] = torch.zeros(1, args.num_classes, H, W)

    # loop of classes
    for oid in range(args.num_classes):
        t = tracks[oid].item() # the first frame step for oid class
        if not t in init_masks: 
            init_masks[t] = mask2tensor(masks_gt[oid], oid)
        else:
            init_masks[t] += mask2tensor(masks_gt[oid], oid)
    
    return init_masks

def make_onehot(mask, HW):
    # convert mask tensor with probabilities to a one-hot tensor
    b,c,h,w = mask.shape

    mask_up = F.interpolate(mask, HW, mode="bilinear", align_corners=True)
    one_hot = torch.zeros_like(mask_up)
    one_hot.scatter_(1, mask_up.argmax(1, keepdim=True), 1)
    one_hot = F.interpolate(one_hot, (h,w), mode="bilinear", align_corners=True)

    return one_hot

def scale_smallest(frame, a):
    H,W = frame.shape[-2:]
    s = a / min(H, W)
    h, w = int(s * H), int(s * W)
    return F.interpolate(frame, (h, w), mode="bilinear", align_corners=True)

def valid_mask(mask):
    """From a tensor [1,C,h,w]
    create [1,C,1,1] 0/1 mask saying which IDs are present"""
    B,C,h,w = mask.shape
   
    valid = mask.flatten(2,3).sum(-1) > 0
    # print(valid.shape) # 1, 7
    valid = valid.type_as(mask).view(B,C,1,1)
    # print(valid.shape) # 1, 7, 1, 1

    return valid

def merge_mask_ids(masks, key0):

    merged_mask = torch.zeros_like(masks[key0])
    for tt, mask in masks.items():
        merged_mask[:,1:] += mask[:,1:]

    probs, ids = merged_mask.max(1, keepdim=True)
    merged_mask.zero_()
    merged_mask.scatter(1, ids, probs)
    merged_mask[:, :1] = 1 - probs
    return merged_mask


@torch.no_grad()
def extract_feature(args, model, frames):
    # get patch embed
    # return: 1, dim, h, w

    if next(model.parameters()).is_cuda:
        out = model.get_intermediate_layers(frames, n=args.test_last_n_layer)[0]
    else: 
        out = model.get_intermediate_layers(frames.cpu(), n=args.test_last_n_layer)[0]

    out = out[:, 1:, :] # B, N, C

    h, w = int(frames.shape[-2] / model.patch_embed.patch_size), int(frames.shape[-1] / model.patch_embed.patch_size)
    dim = out.shape[-1]
    out = out.reshape(-1, h, w, dim)
    embd0 = out.permute(0,3,1,2)

    return embd0


def norm_mask(mask):
    # mask: 7, h, w

    c, h, w = mask.size()

    for cnt in range(c):
        mask_cnt = mask[cnt,:,:]
        if(mask_cnt.max() > 0):
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt/mask_cnt.max()

            mask[cnt,:,:] = mask_cnt

    return mask


def load_mask(cache_dir, size_mask_neighborhood, h, w,):
    # generating mask is time consuming, so we pre-caculate the masks and cache them under cache_dir
    mask_name = 'shape_{}_{}_r_{}.pth'.format(h, w, size_mask_neighborhood)
    print('Loading mask {}'.format(mask_name))
    
    mask_path = os.path.join(cache_dir, mask_name)
    assert os.path.exists(mask_path), print(mask_path)

    mask = torch.load(mask_path)
    print('Loaded mask size is {}'.format(mask.shape))

    return mask


def step_seg(args, labelprop, mask_init, frames, frames_feat=None, net=None):
    
    scale_as = lambda x, y: F.interpolate(x, y.shape[-2:], mode="bilinear", align_corners=True)
    scale = lambda x, hw: F.interpolate(x, hw, mode="bilinear", align_corners=True)

    T = frames.shape[0]

    # ===== TO CUDA ====
    for t in mask_init.keys():
        mask_init[t] = mask_init[t].cuda()
    H, W = mask_init[0].shape[-2:] # H, W is the original image size, the output mask will be 

    ref_embd = {}   # context embeddings
    ref_masks = {}
    ref_valid = {}

    all_masks = [] # class id of each pixel 
    all_masks_conf = [] # confidence 

    # copy from DINO script
    def add_result(mask_pred):
        # input: output mask of labelprop
        # mask_pred: 1, 7, h, w

        # interpolate to original size 
        mask_pred = F.interpolate(mask_pred, scale_factor=args.patch_size, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]

        # one-hot to category
        mask_pred = norm_mask(mask_pred) # NOTE: We recommend to use it for DAVIS, while not for YTVOS.
        _, nxt_masks_id = torch.max(mask_pred, dim=0)

        # resize to original size.
        nxt_masks_id = np.array(nxt_masks_id.squeeze().cpu(), dtype=np.uint8)
        nxt_masks_id = np.array(Image.fromarray(nxt_masks_id).resize((W, H), 0))
        
        all_masks.append(torch.from_numpy(nxt_masks_id).unsqueeze(0))

        return all_masks

    # ====== extract first frame feat ======
    embd0 = extract_feature(args, net, frames[:1].cuda())

    # ====== down scale first seg to embed size =====
    mask0 = scale_as(mask_init[0], embd0)
    
    _, mask0_id = torch.max(mask_init[0].cpu(), dim=1) # 
    all_masks.append(mask0_id)

    ref_embd[0] = {0: embd0} # key mean , value
    ref_masks[0] = {0: mask0}
    ref_valid[0] = valid_mask(mask0) # [x,c,1,1]

    # add this to the reference context
    # if there are objects
    ref_index = []
    if mask_init[0].sum() > 0:
        ref_index = [0]

    # load cached masks (for the sake of efficiency)
    mask_neighborhood = load_mask(args.mask_cache_path, args.size_mask_neighborhood, embd0.shape[-2], embd0.shape[-1], )
    mask_neighborhood = mask_neighborhood.cuda()

    print(">", end='')
    for t in range(1, T):
        print(".", end='')
        sys.stdout.flush()
        frames_batch = frames[t:t+1].cuda()
        nxt_embd = extract_feature(args, net, frames_batch)
        ref_t = [0] if len(ref_index) == 0 else ref_index

        # loop of reference indexes
        nxt_masks = {}
        for t0 in ref_t:
            cxt_index = labelprop.context_index(t0, t) # print(labelprop.context_index(2,10)) # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9]
            cxt_embd = [ref_embd[t0][j] for j in cxt_index]  
            cxt_masks = [ref_masks[t0][j] for j in cxt_index] 
            nxt_masks[t0], mask_neighborhood = labelprop.predict(args, cxt_embd, cxt_masks, nxt_embd, mask_neighborhood)
            torch.cuda.empty_cache()

        # merging all the masks from diffferent reference masks 
        nxt_mask = sum([ref_valid[tt] * nxt_masks[tt] for tt in nxt_masks.keys()])
        #nxt_mask = merge_mask_ids(nxt_masks, ref_t[0]) # 1, 7, 60, 113

        # ===== if there exists new adding labels in current frame ===== 
        if t in mask_init: # not t >= 0
            print("Adding GT mask t = ", t)
            # adding the initial mask if just appeared (replace the GT )
            mask_init_dn = scale_as(mask_init[t], nxt_embd)
            mask_init_dn_s = mask_init_dn.sum(1, keepdim=True)
            nxt_mask = (1 - mask_init_dn_s) * nxt_mask + mask_init_dn_s * mask_init_dn

            # adding to context
            ref_embd[t] = {}
            ref_masks[t] = {}
            ref_valid[t] = valid_mask(mask_init[t])
            ref_index.append(t)

        add_result(nxt_mask)
        ref_t = [0] if len(ref_index) == 0 else ref_index

        # updating the context
        for t0 in ref_t:
            ref_embd[t0][t] = nxt_embd.clone()
            ref_masks[t0][t] = nxt_mask.clone()

            index_short = labelprop.context_long(t0, t)

            tsteps = list(ref_embd[t0].keys())
            for tt in tsteps:
                if t - tt > args.n_last_frames and not tt in index_short:
                    del ref_embd[t0][tt]
                    del ref_masks[t0][tt]

    print('<')
    masks_pred = torch.cat(all_masks, 0)
    # masks_pred_conf = torch.cat(all_masks_conf, 0)
    torch.cuda.empty_cache()
    
    return masks_pred

def extra_args(parser):
    
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'resnet50'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="student", type=str, help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")
    parser.add_argument("--n_last_frames", type=int, default=10, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=40, type=int,
        help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    
    parser.add_argument('--img_size', default=64, type=int, help='Initialized learnable PE size of ViT.')
    parser.add_argument("--test_last_n_layer", type=int, default=6, help="Batch size, try to reduce if OOM")\
    
    parser.add_argument('--mask_cache_path', default='', type=str, help="Path to cached masks.")
    parser.add_argument('--remark', default='', type=str, help="Path to pretrained weights to evaluate.")
    
    parser.add_argument("--scale_size", type=int, default=480, help="scale size of input img, davis (480), ytvos None")
    parser.add_argument("--round_number", type=int, default=8, help="image" )
    
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--mask-output-dir', type=str, default=None, help='inference output dir')
    parser.add_argument("--infer-list", default="voc12/val.txt", type=str)
    parser.add_argument("--resume", type=str, default=None, help="Snapshot \"ID,iter\" to load")
    
    parser.add_argument("--num_classes", type=int, default=7, help="one-hot vector length" )
    
    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="INO")
    parser = extra_args(parser)
    args = parser.parse_args(sys.argv[1:])
    
    if args.scale_size == -1:
        args.scale_size = None
    else:
        args.scale_size = [args.scale_size]
    print('Iamge scale size is {}, Round number is {}'.format(args.scale_size, args.round_number))
    print('Inference parameters are topk {} n_last_frames {} size_mask_neighborhood {} test_last_n_layer {}'.format(args.topk, args.n_last_frames, args.size_mask_neighborhood, args.test_last_n_layer))

    # initialising the dirs
    check_dir(args.mask_output_dir, "vis") # visualization save dir
    check_dir(args.mask_output_dir, "vos") # mask save dir 

    # initialising label propagtion module
    labelprop = LabelPropVOS_DINO(args)

    # initialising model (if using extracted feature, the model is not needed)
    model= vits.__dict__[args.arch](img_size=[args.img_size, args.img_size], patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    vits.load_pretrained_weights(model, args.resume, args.checkpoint_key, args.arch, args.patch_size, filter_keys=True)
    print('Pretrained weights loaded.')
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # check before run for resuming
    infer_list_pth = args.infer_list
    vos_save_dir = os.path.join(args.mask_output_dir, "vos")
    rest_lines = check_before_run(infer_list_pth, vos_save_dir)

    # load dataset 
    dataset = DataSeg(args, rest_lines, num_classes=args.num_classes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, \
                                    drop_last=False) #, num_workers=args.workers)
    palette = dataloader.dataset.get_palette()

    timer = Timer()

    ### Multi-process for 
    print('multi-process initializing ... ')
    pool = mp.Pool(processes=args.workers)
    writer = ResultWriter(davis_palette, args.mask_output_dir)
    print('multi-process initialized')

    # loop of video sequence. 
    for iter, batch in tqdm(enumerate(dataloader)):
        frames_orig, frames, masks_gt, tracks, num_ids, fns, flags, seq_name = batch

        # print(seq_name,) # tuple ('xxxx', )
        # print(fns, ) # list  [('00000',), ('00001',), ('00002',), ('00003',),]
        # print(frames[0].shape) # 69, 3, 480, 910
        print("Sequence {:02d} | {}".format(iter, seq_name[0]))

        # flatten the 1 bacth dim
        masks_gt = masks_gt.flatten(0,1) # mask [7, 480, 910]
        frames_orig = frames_orig.flatten(0,1) # original frames: [69, 3, 480, 910]
        frames = frames.flatten(0,1) # rounded frames: [69, 3, 480, 904]
        tracks = tracks.flatten(0,1) # tacks of the first occurance of each class: [7]
        flags = flags.flatten(0,1) # indicates whether to save the seg results for current frame: [69]

        init_masks = configure_tracks(args, masks_gt, tracks, num_ids[0])

        assert 0 in init_masks, "initial frame has no instances"

        with torch.no_grad():
            masks_pred = step_seg(args, labelprop, init_masks, frames, net=model) # init_masks: {0:(1, 7, h, w), 40:(1, 7, h, w)}, key indicates the frame index of first occurance

        frames_orig = dataset.denorm(frames_orig)

        # save results
        pool.apply_async(writer.save, args=(frames_orig, masks_pred.cpu(), masks_gt.cpu(), flags, fns, seq_name[0]))
        
    timer.stage("Inference completed")
    pool.close()
    pool.join()
