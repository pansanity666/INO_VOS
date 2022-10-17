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
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

import utils
from utils import trunc_normal_

import os 


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads # 8
        head_dim = dim // num_heads # 384 // 8 = 48
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape # 1, 8040, 384
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        
        # atten.shape: 
        
        y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attention:
            return x, attn
        else:
            return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        print('!!! position embedding img size is {}'.format(img_size))
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):

        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class VisionTransformer(nn.Module):
    """ Vision Transforsmer.  """
    # NOTE: We add the mask token implementation on top of the code from DINO -- Xiao Pan. 

    def __init__(self, use_learnable_pos_emb=True, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.use_learnable_pos_emb = use_learnable_pos_emb

        if self.use_learnable_pos_emb:
            print('!!! using learnable position embedding')
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            print('!!! using sincos position embedding')
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
       
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):

        # self.pos_embed: 1, num_patches + 1, embed_dim
        # x: B, num_patches_input + 1, embed_dim
        # w,h: input x size

        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]

        # interpolate patch_pos_embed
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        # print(patch_pos_embed.shape)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


    def prepare_tokens(self, x, mask):
        B, nc, w, h = x.shape

        # print(x.shape) # bs, 3, 256, 256
        x = self.patch_embed(x)  # patch linear embedding
        # print(x.shape) # bs, 1024, 192

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        # print(x.shape) # bs, 1+1024, 192
    
        B, _, C = x.shape
        x_cls = x[:,0,:].unsqueeze(1)
        x_hw = x[:,1:,:]

        if not mask is None and not mask.sum()==0:   # mask.sum()==0 when maskraiot = 0, 
            x_hw[mask] = self.mask_token    
        x = torch.cat([x_cls, x_hw], 1)

        if self.use_learnable_pos_emb:
            x = x + self.interpolate_pos_encoding(x, w, h)
        else:
            x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        return self.pos_drop(x)


    def forward(self, x, mask, return_atten=False):

        x = self.prepare_tokens(x, mask) # 1, 8041, 384

        for i, blk in enumerate(self.blocks):
            
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, atten = blk(x, return_attention=True)

        x = self.norm(x)

        if return_atten:
            return x[:, 0], x[:, 1:], atten.detach()
        else:
            return x[:, 0], x[:, 1:]  
    
    def forward_multilayer(self, x, mask, output_layers, return_atten=False):

        x = self.prepare_tokens(x, mask) # 1, 8041, 384

        ret_cls = []
        ret_hw = []
        for i, blk in enumerate(self.blocks):
            
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, atten = blk(x, return_attention=True)

            if len(self.blocks) - i in output_layers:
                x = self.norm(x)
                ret_cls.append(x[:, 0])
                ret_hw.append(x[:, 1:])
       
        if return_atten:
            return ret_cls, ret_hw, atten.detach()
        else:
            return ret_cls, ret_hw  

    def get_last_selfattention(self, x, mask=None):
        x = self.prepare_tokens(x, mask)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)[1]

    def get_intermediate_layers(self, x, mask=None, n=1):
        x = self.prepare_tokens(x, mask)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output

def vit_tiny(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(img_size=img_size,
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(img_size=img_size,
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base(img_size, patch_size=16, **kwargs):
    model = VisionTransformer(img_size=img_size,
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        
        nlayers = max(nlayers, 1)
        print('!!! DINOHead nlayers is {}'.format(nlayers))
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)

        return x

class MultiLayerWrapper(nn.Module):
    """
    Perform forward pass with multi layer output from ViT as input, and pass each layer to the corresponding Module item in the module list.
    """
    def __init__(self, headlist):
        super(MultiLayerWrapper, self).__init__()

        # disable layers dedicated to ImageNet labels classification
        self.headlist = headlist
    
    def forward(self, x_list):

        # x_list: [(B, C), (B, C), ... ] nlayer items
       
        ret = []
        for head, x in zip(self.headlist, x_list):
            ret.append(head(x))

        return ret

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, args, backbone, headlist, ):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.args = args
        self.backbone = backbone
        self.headlist = MultiLayerWrapper(headlist)

    def forward(self, x, mask=None, return_hw=False, return_atten=False):
        
        # x: [ (bs, 3, h1, w1), (bs, 3, h2, w2)]
        # print(len(x))
        # convert to list

        # if not mask is None: 
        #     assert len(x) == 1
 
        if not isinstance(x, list):
            x = [x]

        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
    
        
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        output_hw_bfr_prj = []
        output_hw_aft_prj = []
        atten_list = []        

        output_bfr_prj = []
        output_afr_prj = []
        
        for end_idx in idx_crops:
            
            # if return_atten:
            #     _out, _out_hw, atten = self.backbone(torch.cat(x[start_idx: end_idx]), return_atten=return_atten)
            #     # print(atten.shape) # B, num_head, 1+h*w, 1+h*w
            #     # assert False
            #     atten_list.append(atten)

            # else:
            #     _out, _out_hw = self.backbone(torch.cat(x[start_idx: end_idx]))
            # # print(_out.shape, _out_hw.shape) # 20, 384; 20, 196, 384   

            if mask == None or mask[start_idx: end_idx] == [None]:
                _mask = None
            else:
                _mask = torch.cat(mask[start_idx: end_idx])

            _out, _out_hw, atten = self.backbone.forward_multilayer(torch.cat(x[start_idx: end_idx]), mask=_mask, output_layers=self.args.multi_scale_layer, return_atten=True)
            
            # print(len(_out), _out[0].shape)
            # print(len(_out_hw), _out_hw[0].shape) # 6; 16, 784, 384
            # assert False

            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
                
            # accumulate outputs
            # output = torch.cat((output, _out))
            start_idx = end_idx
            
            output_bfr_prj.append(_out)
            output_afr_prj.append(self.headlist(_out))

            if return_hw:
                output_hw_bfr_prj.append(_out_hw)
                output_hw_aft_prj.append(self.headlist(_out_hw))

            if return_atten:
                atten_list.append(atten)

        # for at in atten_list:
        #     print(at.shape)
        # assert False

        # print(output.shape) # [128, 384]; [640, 384]
        # Run the head forward on the concatenated features.
        ret_list = []
        # ret_list.extend([output, self.head(output)])
        ret_list.extend([output_bfr_prj, output_afr_prj])

        if return_hw:
            ret_list.extend([output_hw_bfr_prj, output_hw_aft_prj])     

        if return_atten:
            ret_list.extend([atten_list])

        return  ret_list




# kind of ugly here, dont know the better implementation. 
import vision_transformer as vits
def make_model(args, ):
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](img_size=[args.img_size,args.img_size],
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](img_size=[args.img_size,args.img_size], patch_size=args.patch_size)
        embed_dim = student.embed_dim

    # Ignore the repeat_module implementation here. We only supervise the laster layer output of ViT for the final version. 
    DINOHead_list_studnet = utils.get_repeat_module_list(args, DINOHead, len(args.multi_scale_layer), embed_dim, args.out_dim, use_bn=args.use_bn_in_head, norm_last_layer=args.norm_last_layer, nlayers=args.nlayers)
    DINOHead_list_teacher = utils.get_repeat_module_list(args, DINOHead, len(args.multi_scale_layer), embed_dim, args.out_dim, args.use_bn_in_head, nlayers=args.nlayers)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = MultiCropWrapper(args, student, DINOHead_list_studnet)
    teacher = MultiCropWrapper(args, teacher, DINOHead_list_teacher)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        if args.distributed:
            print('!!! Moving teacher model to distributed data parallel ... ')
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True ) # find_unused_parameters=True
        else:
            print('!!! Moving teacher model to data parallel ... ')
            teacher = torch.nn.parallel.DataParallel(teacher)

        teacher_without_ddp = teacher.module

    else: 
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
      
    # move to distrited parallel
    if args.distributed:
        print('!!! Moving student model to distributed data parallel ... ')
        student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu],  find_unused_parameters=True ) # find_unused_parameters=True
    else:
        print('!!! Moving student model to data parallel ... ')
        student = torch.nn.parallel.DataParallel(student)

    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # stop gradients for teacher 
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    return student, teacher, teacher_without_ddp




def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size, filter_keys=False):

    if os.path.isfile(pretrained_weights):

        state_dict = torch.load(pretrained_weights, map_location="cpu")

        # # ===== from VRW ckpt =====
        if 'model' in state_dict:
            state_dict = state_dict['model']
        # remov 'encoder' prefix
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}

        # # ===== from DUL ckpt =====
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # ===== from INO ckpt =====
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        if filter_keys:
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
     
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif model_name == "xcit_small_12_p16":
            url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
        elif model_name == "xcit_small_12_p8":
            url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
        elif model_name == "xcit_medium_24_p16":
            url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
        elif model_name == "xcit_medium_24_p8":
            url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
        elif model_name == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

