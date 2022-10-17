"""
Based on the inference routines from Jabri et al., (2020)
Credit: https://github.com/ajabri/videowalk.git
License: MIT
"""

import sys
import torch
from labelprop.dino import label_propagation

class LabelPropVOS(object):

    def context_long(self):
        """Returns indices of the timesteps
        for long-term memory
        """
        raise NotImplementedError()

    def context_short(self, t):
        """
        Args:
            t: current timestep
        Returns:
            list: indices of timesteps
                  for the context
        """
        raise NotImplementedError()
    
    def predict(self, feats, masks, curr_feat):
        """
        Args:
            feats [C,K,h,w]: context features
            masks [C,M,h,w]: context masks
            curr_feat [1,K,h,w]: current frame features
        Returns:
            mask [1,M,h,w]: current frame mask
        """
        raise NotImplementedError()




class LabelPropVOS_DINO(LabelPropVOS):

    def __init__(self, args):
        self.cxt_size = args.n_last_frames

    def context_long(self, t0, t):
        return [t0]

    def context_short(self, t0, t):
        # last cxr_size indexes
        to_t = t
        from_t = to_t - self.cxt_size
        timesteps = [max(tt, t0) for tt in range(from_t, to_t)]
        timesteps = [ _t for _t in timesteps if not _t == t0] # filter the duplicated 0 
        return timesteps

    def context_index(self, t0, t):
        # t0: start index of reference mask 
        # t: current frame index
        index_short = self.context_short(t0, t)
        index_long = self.context_long(t0, t)
        cxt_index = index_long + index_short
        return cxt_index

    def predict(self, args, feats, masks, curr_feat, mask_neighborhood):
        """
        Args:
            feats: list of C [1,K,h,w] context features
            masks: list of C [1,M,h,w] context masks
            curr_feat: [1,K,h,w] current frame features
            ref_index: C indices of context frames
            t: current frame time step
        Returns:
            mask [1,M,h,w]: current frame mask
        """
        mask_cur, mask_neighborhood = label_propagation(args, feats, curr_feat, masks, mask_neighborhood, use_efficient=True)
        return mask_cur, mask_neighborhood

       



