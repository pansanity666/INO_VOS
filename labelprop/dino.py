

import os
import copy
import glob
import queue
import argparse
import numpy as np
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms

def restrict_neighborhood(args, h, w, is_cuda=True):
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
  
    for i in range(h):
        for j in range(w):
            for p in range(2 * args.size_mask_neighborhood + 1):
                for q in range(2 * args.size_mask_neighborhood + 1):
                    if i - args.size_mask_neighborhood + p < 0 or i - args.size_mask_neighborhood + p >= h:
                        continue
                    if j - args.size_mask_neighborhood + q < 0 or j - args.size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - args.size_mask_neighborhood + p, j - args.size_mask_neighborhood + q] = 1

                    # print(i,j,p,q)
    mask = mask.reshape(h * w, h * w)

    return mask.cuda(non_blocking=True) if is_cuda else mask


def label_propagation(args, feats, curr_feat, masks, mask_neighborhood=None, use_cpu=False, use_efficient=False):
    """
    propagate segs of frames in list_frames to frame_tar
    """

    # feat_tar.shape: 1, dim, h*w
    # feat_sources.shape: nm_context, dim, h*w
    # list_segs: list of reference masks

    _, K, h, w = curr_feat.shape
    feats = [ f.flatten(-2,-1)  for f in feats] # 1, K, h, w -> 1, K, h*w

    # for f in feats:
        # print(f.shape) # 1, 384, 6780
    # print(curr_feat.shape) # 1, 384, 60, 113

    ncontext = len(feats)
    feat_sources = torch.cat(feats) # n_context, dim, h*w
    curr_feat = curr_feat.flatten(-2,-1) # 1, dim, h*w

    ncontext = feat_sources.shape[0]
    feat_tar = curr_feat[0].T # h*w, dim
    # print(feat_tar.shape, feat_sources.shape) # 6780, 384; 8, 1, 384, 6780
   
    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2) # NC, dim, h*w
    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1) # NC, h*w, dim

    if use_efficient:
        
        # prepare segs 
        masks = [s.cuda() for s in masks]
        segs = torch.cat(masks)
        nmb_context, C, h, w = segs.shape
        segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C, NC*h*w

        # batch query forward
        seg_tar = []
        tar_bs = 100
        for bs in range(0, feat_tar.shape[1], tar_bs):
            _feat_tar = feat_tar[:,bs:bs+tar_bs,:] # NC, 100, dim

            cur_bs = _feat_tar.shape[1]

            _aff = torch.exp(torch.bmm(_feat_tar, feat_sources) / 0.1) # NC, 100, dim; NC, dim, h*w
            _aff *= mask_neighborhood[bs:bs+tar_bs] # NC, 100, h*w
            _aff = _aff.transpose(2, 1).reshape(-1, cur_bs)  # NC*h*w, 100


            tk_val, _ = torch.topk(_aff, dim=0, k=args.topk)
            tk_val_min, _ = torch.min(tk_val, dim=0)
            
            _aff[_aff < tk_val_min] = 0
            _aff = _aff / torch.sum(_aff, keepdim=True, axis=0) # NC*h*w, 100

            _seg_tar = torch.mm(segs, _aff) # C, 100
            seg_tar.append(_seg_tar)
        
        seg_tar = torch.cat(seg_tar, dim=-1) # C, h*w
        seg_tar = seg_tar.reshape(1, C, h, w)

        return seg_tar, mask_neighborhood

    else:
        # NOTE: calculate bmm on GPU
        aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1) # nmb_context x h*w (tar: query) x h*w (source: keys)
        
        assert mask_neighborhood is not None

        aff *= mask_neighborhood
        aff = aff.transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source: keys) x h*w (tar: queries)

        tk_val, _ = torch.topk(aff, dim=0, k=args.topk)
        tk_val_min, _ = torch.min(tk_val, dim=0)

        if use_cpu:
            aff = aff.cpu()
            tk_val_min = tk_val_min.cpu()
            aff[aff < tk_val_min] = 0
            aff = aff.cuda()
        else:
            aff[aff < tk_val_min] = 0

        aff = aff / torch.sum(aff, keepdim=True, axis=0)
       
        # prepare segs 
        masks = [s.cuda() for s in masks]
        segs = torch.cat(masks)
        nmb_context, C, h, w = segs.shape
        segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
        # propogate segs
        seg_tar = torch.mm(segs, aff)
        seg_tar = seg_tar.reshape(1, C, h, w)
    
    # print(seg_tar.shape) # 1, 7, 60, 113
    return seg_tar, mask_neighborhood