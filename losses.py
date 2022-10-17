import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import utils


class DINOLoss(nn.Module):
    # We use DINOLoss for calculation out_g2g and out_l2g.
    def __init__(self, args, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()

        self.args = args
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        # self.ncrops = ncrops    # 8 + 2views
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, n_gloabl=2, n_local=8, seq_length=2, args=None, return_entropy_kl=True, local2glb_only=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # if local2glb_only = True, only out_l2g is caculated. 

        assert student_output.requires_grad==True and teacher_output.requires_grad==False

        # print(student_output.ndim)
        assert student_output.ndim == 2
        assert teacher_output.ndim == 2

        # print(n_gloabl, n_local) 2, 16
        # print(teacher_output.shape) # 4, 4096
        # print(student_output.shape) # 36, 4096

        student_out = student_output / self.student_temp
        # print(student_out.shape) # 640, K

        if local2glb_only:
            student_out = student_out.chunk((n_local)*seq_length)
        else:
            student_out = student_out.chunk((n_gloabl + n_local)*seq_length)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(n_gloabl*seq_length) # global * args.seq_length

        if local2glb_only:
            total_loss, loss_entropy, loss_KL = self.local2glb(teacher_out, student_out)
        else:
            total_loss = self.glb_local2glb(teacher_out, student_out)

        self.update_center(teacher_output)

        if return_entropy_kl:
            return total_loss, loss_entropy.detach(), loss_KL.detach()
        else:
            return total_loss

    def local2glb(self, teacher_out, student_out):
        # student_out contains local ouputs only
        
        total_loss_ce = 0
        total_loss_entropy = 0
        total_loss_KL = 0

        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                
                loss_ce = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean()
                loss_entropy = torch.sum(-q * F.log_softmax(q, dim=-1), dim=-1).mean()
                loss_KL = -loss_entropy + loss_ce

                total_loss_ce += loss_ce
                total_loss_entropy += loss_entropy
                total_loss_KL += loss_KL
                n_loss_terms += 1
        
        return total_loss_ce/n_loss_terms, total_loss_entropy/n_loss_terms, total_loss_KL/n_loss_terms


    def glb_local2glb(self, teacher_out, student_out):
        # student_out contains glb + local outputs

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        return total_loss/n_loss_terms


    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)

        # print(teacher_output.shape)
        # print(batch_center.shape)
        # assert False

        if self.args.distributed:
            dist.all_reduce(batch_center) # sum all batch center in different cards. 
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size()) # divide all batch_size: syncronized center
        else:
            batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

def calc_cross_frame_dino(args, glb_teacher_frame, glb_student_frame, local_student_frame, dino_loss, epoch):
    # zip [f0, f1, f2, f3] to [(f0, f2), (f1, f3)], and then calculation DINO Loss between each frame pairt. 
    # This loss is named as out_g2g + out_l2g in the paper. 

    T, C = glb_teacher_frame.shape[0], glb_teacher_frame.shape[-1]
    ng, nl = args.global_crops_number, args.local_crops_number
    B = args.batch_size_per_gpu
    assert T%2 == 0
    if nl == 0: assert local_student_frame is None

    glb_teacher_frame = glb_teacher_frame.view(2, int(T//2), ng, B, C).permute(1,0,2,3,4).flatten(1,3) # T//2, 2*ng*B, C
    glb_student_frame = glb_student_frame.view(2, int(T//2), ng, B, C).permute(1,0,2,3,4).flatten(1,3)
    if not nl==0:
        local_student_frame = local_student_frame.view(2, int(T//2), nl, B, C).permute(1,0,2,3,4).flatten(1,3) # T//2, 2*nl*B, C

    loss_cfdino = torch.tensor(0.).cuda()
    for t in range(int(T//2)):
        if  nl > 0:
            dino_student_input = torch.cat([glb_student_frame[t], local_student_frame[t]], dim=0) # (2*ng*B + 2*nl*B), C
        else:
            dino_student_input = glb_student_frame[t] # 2*ng*B, C
        dino_teacher_input = glb_teacher_frame[t]  # 2*ng*B, C
        loss_cfdino += dino_loss(dino_student_input, dino_teacher_input, epoch, n_gloabl=2*ng, n_local=2*nl, seq_length=1, args=args, return_entropy_kl=False, local2glb_only=False)
    loss_cfdino /= int(T//2)

    return loss_cfdino

def calc_mim_loss(args, glb_student_hw, glb_teacher_hw, mask_frame, dino_loss_hw, epoch):

    # glb_teacher_hw: T, ng*B, N, C
    # glb_student_hw: T, ng*B, N, C
    # mask_frame: T, ng, B, N

    B = args.batch_size_per_gpu
    T, _, _, C = glb_teacher_hw.shape

    loss_mim = torch.tensor(0.).cuda()
    mask_cnt = 0
    # loop of T
    for i in range(T):
        # mask_frame[i] # B, N
        mask = mask_frame[i]

        if mask.sum() == 0: # mask_ratio=0
            pass 
        else:
            student_hw_mask = glb_student_hw[i][mask.flatten(0,1)].reshape(B, -1, C) # B, nummask, C
            teacher_hw_mask = glb_teacher_hw[i][mask.flatten(0,1)].reshape(B, -1, C) # B, nummask, C
            loss_mim += dino_loss_hw(student_hw_mask.flatten(0,1), teacher_hw_mask.flatten(0,1), epoch, 
                                    n_gloabl=args.global_crops_number, n_local=args.global_crops_number, 
                                    seq_length=1, args=args, 
                                    return_entropy_kl=False, local2glb_only=True)
            mask_cnt += 1

    if mask_cnt:
        loss_mim = loss_mim/mask_cnt

    return loss_mim



def calc_affine(args, mask_frame, affine_teacher_temp, glb_student_hw, glb_teacher_hw):

    T = args.seq_length
    ng = args.global_crops_number
    B = args.batch_size_per_gpu
    # get TBNC
    TBNC_student = glb_student_hw.view(T,ng,B,*glb_student_hw.shape[-2:]).flatten(0,1) # T*ng, B, N, C
    TBNC_teacher = glb_teacher_hw.view(T,ng,B,*glb_teacher_hw.shape[-2:]).flatten(0,1) 
    # only affine on mask tokens
    # mask_frame: T, ng, B, N
    T, B, _, C = TBNC_student.shape
    _mask_frame = mask_frame.flatten(0,1) # T*ng, B, N_
    # print(_mask_frame.shape, TBNC_student.shape) # 4, 3, 784; 4,3,784,4096
    TBNC_student = TBNC_student[_mask_frame].view(T, B, -1, C) # 
    TBNC_teacher = TBNC_teacher[_mask_frame].view(T, B, -1, C) # 
    # TBNC -> BCTN
    BCTN_student = TBNC_student.permute(1,3,0,2)
    BCTN_teacher = TBNC_teacher.permute(1,3,0,2)
    # normalize BCTN
    BCTN_student = F.normalize(BCTN_student, p=2, dim=1, ) # Important
    BCTN_teacher = F.normalize(BCTN_teacher, p=2, dim=1, ) # Important
    # get affine matrix
    BTNN_student = utils.affinity(BCTN_student[:,:,:-1], BCTN_student[:,:,1:]) # B, T-1, N, N
    BTNN_teacher = utils.affinity(BCTN_teacher[:,:,:-1], BCTN_teacher[:,:,1:]) # B, T-1, N, N
    # CE loss
    BTNN_teacher, BTNN_student = BTNN_teacher.flatten(0,2), BTNN_student.flatten(0,2)
    BTNN_teacher = F.softmax(BTNN_teacher/affine_teacher_temp, dim=-1)
    BTNN_student = F.log_softmax(BTNN_student/args.affine_student_temp, dim=-1)
    loss_affine = torch.sum(-BTNN_teacher * BTNN_student, dim=-1).mean()
    
    return loss_affine