#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   criterion.py
@Time    :   8/30/19 8:59 PM
@Desc    :
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""
import einops
import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
from part.lovasz_softmax import LovaszSoftmax
# from part.kl_loss import KLDivergenceLoss
# from .consistency_loss import ConsistencyLoss

class ConsistencyLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(ConsistencyLoss, self).__init__()
        self.ignore_index=ignore_index

    def forward(self, parsing, edge, label):
        parsing_pre = torch.argmax(parsing, dim=1)
        parsing_pre[label==self.ignore_index]=self.ignore_index
        generated_edge = generate_edge_tensor(parsing_pre)
        edge_pre = torch.argmax(edge, dim=1)
        v_generate_edge = generated_edge[label!=255]
        v_edge_pre = edge_pre[label!=255]
        v_edge_pre = v_edge_pre.type(torch.cuda.FloatTensor)
        positive_union = (v_generate_edge==1)&(v_edge_pre==1) # only the positive values count
        l = F.smooth_l1_loss(v_generate_edge[positive_union].squeeze(0), v_edge_pre[positive_union].squeeze(0))
        return l


def flatten_probas(input, target, labels, ignore=255):
    """
    Flattens predictions in the batch.
    """
    B, C, H, W = input.size()
    input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    target = target.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return input, target
    valid = (labels != ignore)
    vinput = input[valid.nonzero().squeeze()]
    vtarget = target[valid.nonzero().squeeze()]
    return vinput, vtarget


class KLDivergenceLoss(nn.Module):
    def __init__(self, ignore_index=255, T=1):
        super(KLDivergenceLoss, self).__init__()
        self.ignore_index=ignore_index
        self.T = T

    def forward(self, input, target, label):
        log_input_prob = F.log_softmax(input / self.T, dim=1)
        target_porb = F.softmax(target / self.T, dim=1)
        loss = F.kl_div(*flatten_probas(log_input_prob, target_porb, label, ignore=self.ignore_index))
        return self.T*self.T*loss # balanced



class CriterionAll(nn.Module):
    def __init__(self, use_class_weight=False, ignore_index=255, lambda_1=1, lambda_2=1, lambda_3=0.1,
                 num_classes=20):
        super(CriterionAll, self).__init__()
        self.ignore_index = ignore_index
        self.use_class_weight = use_class_weight
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.lovasz = LovaszSoftmax(ignore_index=ignore_index)
        self.kldiv = KLDivergenceLoss(ignore_index=ignore_index)
        self.reg = ConsistencyLoss(ignore_index=ignore_index)
        self.lamda_1 = lambda_1
        self.lamda_2 = lambda_2
        self.lamda_3 = lambda_3
        self.num_classes = num_classes

    def parsing_loss(self, preds, target, cycle_n=None):
        """
        Loss function definition.

        Args:
            preds: [[parsing result1, parsing result2],[edge result]]
            target: [parsing label, egde label]
            soft_preds: [[parsing result1, parsing result2],[edge result]]
        Returns:
            Calculated Loss.
        """
        h, w = target[0].size(1), target[0].size(2)

        pos_num = torch.sum(target[1] == 1, dtype=torch.float)
        neg_num = torch.sum(target[1] == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])  # edge loss weight

        loss = 0

        # loss for segmentation
        preds_parsing = preds[0]
        for pred_parsing in preds_parsing:
            scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += 0.5 * self.lamda_1 * (self.lovasz(scale_pred, target[0]) + self.criterion(scale_pred, target[0]))

        # loss for edge
        preds_edge = preds[1]
        for pred_edge in preds_edge:
            scale_pred = F.interpolate(input=pred_edge, size=(h, w),
                                       mode='bilinear', align_corners=True)
            loss += self.lamda_2 * F.cross_entropy(scale_pred, target[1],
                                                       weights.cuda(), ignore_index=self.ignore_index)

        # consistency regularization
        preds_parsing = preds[0]
        preds_edge = preds[1]
        for pred_parsing in preds_parsing:
            scale_pred = F.interpolate(input=pred_parsing, size=(h, w),
                                       mode='bilinear', align_corners=True)
            scale_edge = F.interpolate(input=preds_edge[0], size=(h, w),
                                       mode='bilinear', align_corners=True)
            r = self.reg(scale_pred, scale_edge, target[0])
            if not torch.isnan(r):
                loss += self.lamda_3 * r

        return loss

    def forward(self, preds, target, cycle_n=None):
        loss = self.parsing_loss(preds, target, cycle_n)
        return loss

    def _generate_weights(self, masks, num_classes):
        """
        masks: torch.Tensor with shape [B, H, W]
        """
        masks_label = masks.data.cpu().numpy().astype(np.int64)
        pixel_nums = []
        tot_pixels = 0
        for i in range(num_classes):
            pixel_num_of_cls_i = np.sum(masks_label == i).astype(np.float)
            pixel_nums.append(pixel_num_of_cls_i)
            tot_pixels += pixel_num_of_cls_i
        weights = []
        for i in range(num_classes):
            weights.append(
                (tot_pixels - pixel_nums[i]) / tot_pixels / (num_classes - 1)
            )
        weights = np.array(weights, dtype=np.float)
        # weights = torch.from_numpy(weights).float().to(masks.device)
        return weights


def moving_average(target1, target2, alpha=1.0):
    target = 0
    target += (1.0 - alpha) * target1
    target += target2 * alpha
    return target


def to_one_hot(tensor, num_cls, dim=1, ignore_index=255):
    b, h, w = tensor.shape
    tensor[tensor == ignore_index] = 0
    onehot_tensor = torch.zeros(b, num_cls, h, w).cuda()
    onehot_tensor.scatter_(dim, tensor.unsqueeze(dim), 1)
    return onehot_tensor


def generate_edge_tensor(label, edge_width=3):
    label = label.type(torch.cuda.FloatTensor)
    if len(label.shape) == 2:
        label = label.unsqueeze(0)
    n, h, w = label.shape
    edge = torch.zeros(label.shape, dtype=torch.float).cuda()
    # right
    edge_right = edge[:, 1:h, :]
    edge_right[(label[:, 1:h, :] != label[:, :h - 1, :]) & (label[:, 1:h, :] != 255)
               & (label[:, :h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :, :w - 1]
    edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])
            & (label[:, :, :w - 1] != 255)
            & (label[:, :, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:, :h - 1, :w - 1]
    edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                 & (label[:, :h - 1, :w - 1] != 255)
                 & (label[:, 1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:, :h - 1, 1:w]
    edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                     & (label[:, :h - 1, 1:w] != 255)
                     & (label[:, 1:h, :w - 1] != 255)] = 1

    kernel = torch.ones((1, 1, edge_width, edge_width), dtype=torch.float).cuda()
    with torch.no_grad():
        edge = edge.unsqueeze(1)
        edge = F.conv2d(edge, kernel, stride=1, padding=1)
    edge[edge!=0] = 1
    edge = edge.squeeze()
    return edge


def contrastive_loss(feats, t=0.07):
    feats = F.normalize(feats, dim=2)  # B x K x C
    scores = torch.einsum('aid, bjd -> abij', feats, feats)
    scores = einops.rearrange(scores, 'a b i j -> (a i) (b j)')

    # positive logits: Nx1
    pos_idx = einops.repeat(torch.eye(feats.size(1), dtype=torch.int, device=feats.device), 'i j -> (a i) (b j)', a=feats.size(0), b=feats.size(0))
    pos_idx.fill_diagonal_(0)
    l_pos = torch.gather(scores, 1, pos_idx.nonzero()[:, 1].view(scores.size(0), -1))
    # rand_idx = torch.randint(1, l_pos.size(1), (l_pos.size(0), 1), device=feats.device)
    l_pos, _ = l_pos.min(dim=1) #torch.gather(l_pos, 1, rand_idx)

    # negative logits: NxK
    neg_idx = einops.repeat(1-torch.eye(feats.size(1), dtype=torch.int, device=feats.device), 'i j -> (a i) (b j)', a=feats.size(0), b=feats.size(0))
    l_neg = torch.gather(scores, 1, neg_idx.nonzero()[:, 1].view(scores.size(0), -1))
    # logits: Nx(1+K)
    logits = torch.cat([l_pos.unsqueeze(1), l_neg], dim=1)

    # apply temperature
    logits /= t

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(logits, labels)
