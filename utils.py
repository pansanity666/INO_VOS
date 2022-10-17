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
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
from torchvision import transforms

import errno
from tqdm import tqdm



def affinity(x1, x2):
    
    # input:  x1: B, C, T, N; 
    #         x2: B, C, T, N
    # output: B, T, N, N

    in_t_dim = x1.ndim
    if in_t_dim < 4:  # add in time dimension if not there
        x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)
    A = torch.einsum('bctn,bctm->btnm', x1, x2)

    return A.squeeze(1) if in_t_dim < 4 else A
    
def get_repeat_module_list(args, module, N, *args_, **kwargs):
    
    if 'shareMS' in args.remark:

        model = module(*args_, **kwargs)
        ret = nn.ModuleList()
        for _ in range(N):
            ret.append(model)

    else:

        ret = nn.ModuleList()
        for _ in range(N):
            ret.append(module(*args_, **kwargs))

    return ret


def list2tensor(global_frames):
    # return: seq_len, num_glb, bs, xxxx
    if len(global_frames) == 0 or len(global_frames[0]) == 0:
        return None

    ret = []
    for frame in global_frames:
        temp = []
        for crop in frame:
            temp.append(crop)
        
        ret.append(torch.stack(temp))
    return torch.stack(ret)



def tensor2list(tensor):
    # change the outer dimension of tensor to list 
    return [ t for t in tensor ]


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def chunk_list(arr, n):
    # n: the length of each chunk 
    assert len(arr)%n==0
    
    return [arr[i:i+n] for i in range(0, len(arr), n)]

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


toPIL = transforms.ToPILImage()
toTensor = transforms.ToTensor()
def batch_tensor_to_PIL_list(tensors):

    PILs = []
    for t in tensors:
        PILs.append(toPIL(t))
    
    return PILs

def PIL_list_to_batch_tensor(PILs):

    tensors = []
    for PIL in PILs:
        tensors.append(toTensor(PIL))
    
    return torch.stack(tensors)

class Batch_Solarization(object):

    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        # input: B,C,H,W tensor
        # output: B,C,H,W tensor

        if random.random() < self.p:

            img_PIL = batch_tensor_to_PIL_list(img)
            img_PIL = [ ImageOps.solarize(img)  for img in img_PIL]

            return PIL_list_to_batch_tensor(img_PIL)

        else:

            return img

class Batch_GaussianBlur(object):
    
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        # input: batch tensor
        # output: batch tensor

        do_it = random.random() <= self.prob

        if not do_it:
            return img

        radius = random.uniform(self.radius_min, self.radius_max)

        img_PIL = batch_tensor_to_PIL_list(img)

        img_PIL = [ img.filter(ImageFilter.GaussianBlur(radius=radius))  for img in img_PIL]

        return PIL_list_to_batch_tensor(img_PIL)


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size, filter_keys=False):
    


    if os.path.isfile(pretrained_weights):
        print(1123)
        # assert 1==2
        state_dict = torch.load(pretrained_weights, map_location="cpu")

        # for k in model.state_dict():
        #     print(k)

        # print('+++++')
        # for k in state_dict:
        #     print(k)
        # assert 1==2

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


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule





def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                '{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                '{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())),
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)

    dist.barrier()
    
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


class PCA():
    """
    Class to  compute and apply PCA.
    """
    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][:self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1. / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd) # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos  = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:,i], qgndj)]

        k = 0;
        ij = 0;
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1 # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j]); 
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v


def generate_attention_map(image_pth, model, size, epoch, args, threshold=0.6):

    # read img and generate its attention map
    # image_pth: the path of the original img
    # threshold: reserved percentage
    
    import dataset.YTVOSTrainLoader as YL
    import matplotlib.pyplot as plt
    from torchvision import transforms
    image_inf = YL.PIL_image_loader(image_pth)

    def make_divisible(img):
        w, h = (
            img.shape[1] - img.shape[1] % args.patch_size,
            img.shape[2] - img.shape[2] % args.patch_size,
        )
        img = img[:, :w, :h].unsqueeze(0)       
        return img

    normalize = transforms.Compose([    
        transforms.Resize(size),  # 224, 224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image_inf = make_divisible(normalize(image_inf))
    
    # print(image_inf.shape) # 1, 3, 720, 1280
    h_featmap = image_inf.shape[-2] // args.patch_size
    w_featmap = image_inf.shape[-1] // args.patch_size
    attentions = model.backbone.get_last_selfattention(image_inf.cuda())
    # print(attentions.shape) # 1, 6, 3601, 3601

    nh = attentions.shape[1]  # number of head
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    # print(attentions.shape) # 6, 3600

    # we keep only a certain percentage of the mass
    def filter_by_threshold(attentions, threshold):

        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]

        return th_attn

    
    # upsample with threhold
    th_attn = filter_by_threshold(attentions, threshold)
    th_attn = th_attn.reshape(nh, h_featmap, w_featmap).float()
    th_attn = (
        nn.functional.interpolate(
            th_attn.unsqueeze(0),
            scale_factor=args.patch_size,
            mode="nearest",
        )[0]
        .cpu()
        .numpy()
    )

    # upsample no threhold
    attentions = attentions.reshape(nh, h_featmap, w_featmap)
    attentions = (
        nn.functional.interpolate(
            attentions.unsqueeze(0),
            scale_factor=args.patch_size,
            mode="nearest",
        )[0]
        .cpu()
        .numpy()
    ) 
    

    return  image_inf, torch.from_numpy(attentions), torch.from_numpy(th_attn)

def _sample_index(x, N):
    """Sample indices of the anchors
    Args:
        x: [B,K,H,W]
    Returns:
        index: [B,N*N,K]
    """

    B,K,H,W = x.shape
    # B = x.view(-1,T,K,H*W).shape[0]

    # sample indices from a uniform grid
    xs, ys = W // N, H // N
    x_sample = torch.arange(0, W, xs).view(1, 1, N)
    y_sample = torch.arange(0, H, ys).view(1, N, 1)

    # Random offsets
    # [B x 1 x N]
    x_sample = x_sample + torch.randint(0, xs, (B, 1, 1))
    # [B x N x 1]
    y_sample = y_sample + torch.randint(0, ys, (B, 1, 1))


    # batch index
    # [B x N x N]
    hw_index = torch.LongTensor(x_sample + y_sample * W)
    
  
    return hw_index

def _sample_from(x, index,  N):
    """Gather the features based on the index
    Args:
        x: [B,K,H,W]
        index: [B,N,N] defines the indices of NxN grid for a single
                        frame in each of B video clips
    Returns:
        anchors: [B,NN,K] sampled features given by index from x
    """

    B,K,H,W = x.shape
    # x = x.view(-1,T,K,H*W)
    # B = x.shape[0]

    # > [B,T,K,HW] > [B,T,HW,K] > [B,THW,K]
    x = x.permute([0,1,3,2]).reshape(B,-1,K)

    # every video clip will have the same samples
    # on the grid
    # [B x N x N] -> [B x N*N x 1] -> [B x N*N x K]
    index = index.view(B,-1,1).expand(-1,-1,K)

    # selecting from the uniform grid
    y = x.gather(1, index.to(x.device))

    return y 
    # print(y.shape)
    # # [BNN,K]
    # return y.flatten(0,1)
    
    
    

def get_vid2seg_number(infer_list_pth):

    vid2seg_number = defaultdict(int)
    line_list = []
    with open(infer_list_pth, 'r') as lines:
        for line in tqdm(lines):

            line_list.append(line)

            _flag, _image, _mask = line.strip("\n").split(' ')

            vid_name = _image.split('/')[-2]
            if int(_flag) == 1:
                vid2seg_number[vid_name] += 1

    return vid2seg_number, line_list


def check_existed_videos(vos_save_dir, vid2seg_number):

    from PIL import Image

    failed_vid_list = []
    succeed_vid_list = []

    for vn in os.listdir(vos_save_dir):

        vid_pth = os.path.join(vos_save_dir, vn)

        # try to read every frame 
        # empty file 
        if len(sorted(os.listdir(vid_pth)))==0:
            failed_vid_list.append(vn)
            continue
        
        for img_n in sorted(os.listdir(vid_pth)):

            img_pth = os.path.join(vid_pth, img_n)
            success = 0
            try:
                Image.open(img_pth)
                success = 1
            except Exception as e :
                print(repr(e))

            if not success:
                print('Failed when reading segs {}! Adding vid {} to failed list'.format(img_pth, vn))
                failed_vid_list.append(vn)
                break

        if not success: continue

        # check the total length    
        if not len(sorted(os.listdir(vid_pth))) == vid2seg_number[vn]:
            print('Incomplete Seg numbers! Adding vid {} to failed list, expected {}, in fact {}'.format(vn, vid2seg_number[vn], len(sorted(os.listdir(vid_pth)))))
            failed_vid_list.append(vn)
            continue
            
        succeed_vid_list.append(vn)
        

    print('Totally {} videos, {} succeed, {} failed'.format(len(vid2seg_number.keys()), len(succeed_vid_list), len(failed_vid_list), )) 

    return succeed_vid_list, failed_vid_list



def delete_failed_vid(vos_save_dir, failed_vid_list):

    import shutil

    # print(failed_vid_list)
    for vn in failed_vid_list:
        vn_pth = os.path.join(vos_save_dir, vn)

        # os.removedirs(vn_pth)
        shutil.rmtree(vn_pth)
        print(vn_pth, 'removed!')



def check_before_run(infer_list_pth, vos_save_dir):

    vid2seg_number, full_lines = get_vid2seg_number(infer_list_pth)

    succeed_vid_list, failed_vid_list = check_existed_videos(vos_save_dir, vid2seg_number)

    delete_failed_vid(vos_save_dir, failed_vid_list)

    # get new line list for the rest of videos to be evaluated 
    rest_lines = []
    rest_vid_cnt = defaultdict(int)
    for line in full_lines:

        _flag, _image, _mask = line.strip("\n").split(' ')
        vid_name = _image.split('/')[-2]

        if vid_name not in succeed_vid_list:
            rest_vid_cnt[vid_name] += 1
            rest_lines.append(line)

        else:
            continue

    print('Rest video number {}'.format(len(rest_vid_cnt.keys())))
   
    return rest_lines

def check_dir(base_path, name):
    """Make sure the directory exists"""

    # create the directory
    fullpath = os.path.join(base_path, name)
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    return fullpath

class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))


    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)

    def stage(self, msg=None):
        t = self.get_stage_elapsed()
        self.reset_stage()
        if not msg is None:
            print("{:4.3f} elapsed: {}".format(t, msg))
        return t

    def str_est_finish(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()


class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]
