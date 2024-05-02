import copy
import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

from models.resnet import resnet50, resnet18
from part.losses import contrastive_loss, CPMLoss
from part.part_model import PartModel, DEE_module, PRM_module
from part.transformer import SimpleViT
from utils.calc_acc import calc_acc

from layers import TripletLoss
from layers import CSLoss
from layers import CenterLoss
from layers import cbam
from layers import NonLocalBlockND
from layers import DualBNNeck
from layers.module.part_pooling import TransformerPool, SAFL
from part import part_model
import einops


class Baseline(nn.Module):
    def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False,
                 modality_attention=0, mutual_learning=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.pattern_attention = pattern_attention
        self.modality_attention = modality_attention
        self.mutual_learning = mutual_learning

        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride,
                                     modality_attention=modality_attention)
            D = 2048
        elif backbone == "resnet18":
            self.backbone = resnet18(pretrained=True, drop_last_stride=drop_last_stride,
                                     modality_attention=modality_attention)
            D = 512

        self.v_backbone = copy.deepcopy(self.backbone)
        self.i_backbone = copy.deepcopy(self.backbone)
        self.vi_classifier = nn.Linear(D, 2 * num_classes, bias=False)
        self.v_neck = nn.BatchNorm1d(D)
        self.i_neck = nn.BatchNorm1d(D)
        self.proj = nn.Parameter(torch.zeros([D, D], dtype=torch.float32, requires_grad=True))
        nn.init.kaiming_normal_(self.proj, nonlinearity="linear")
        self.mse_loss = nn.MSELoss()

        self.base_dim = D
        self.dim = D
        self.k_size = kwargs.get('k_size', 8)
        self.part_num = 0 #kwargs.get('num_parts', 7)
        self.dp = kwargs.get('dp', "l2")
        self.dp_w = kwargs.get('dp_w', 0.5)
        self.cs_w = kwargs.get('cs_w', 1.0)
        self.margin1 = kwargs.get('margin1', 0.01)
        self.margin2 = kwargs.get('margin2', 0.7)

        # self.attn_pool = SAFL(part_num=self.part_num)
        self.bn_neck = DualBNNeck(self.base_dim + self.dim * self.part_num)

        self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data
        self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

        self.KL_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.weight_KL = kwargs.get('weight_KL', 2.0)
        self.update_rate = kwargs.get('update_rate', 0.2)
        self.update_rate_ = self.update_rate

        self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.cs_loss_fn = CSLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2)

    def forward(self, inputs, labels=None, **kwargs):
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        # CNN
        global_feat, x3, x2, x1 = self.backbone(inputs)

        v_feat, _, _, _ = self.v_backbone(inputs[sub == 0])
        i_feat, _, _, _ = self.v_backbone(inputs[sub == 1])
        v_feat = v_feat.mean(dim=(2, 3))
        i_feat = i_feat.mean(dim=(2, 3))


        b, c, w, h = global_feat.shape

        # part_feat, attn = self.attn_pool(global_feat)
        global_feat = global_feat.mean(dim=(2, 3))
        feats = global_feat

        if not self.training:
            feats = self.bn_neck(feats, sub)
            return feats, feats
        else:
            return self.train_forward(feats, labels, 0, sub, v_feat, i_feat, **kwargs)

    def train_forward(self, featA, labels, loss_dp, sub, v_feat, i_feat, **kwargs):
        metric = {}

        v_feat = self.v_neck(v_feat)
        i_feat = self.v_neck(i_feat)
        featVI = torch.cat([v_feat, i_feat], 0)
        logits_vi = self.vi_classifier(featVI)
        labelsVI = torch.cat([2 * labels[sub == 0], 2 * labels[sub == 1] + 1], 0)
        # labelsVI[sub == 0] = 2 * labels[sub == 0]
        # labelsVI[sub == 1] = 2 * labels[sub == 1] + 1
        loss_idVI = self.ce_loss_fn(logits_vi.float(), labelsVI)
        proj_inner = torch.mm(F.normalize(self.projs, 2, 0).t(), F.normalize(self.projs, 2, 0))
        eye_label = torch.eye(self.projs.shape[1], device=v_feat.device)
        loss_ortho = (proj_inner - eye_label).abs().sum(1).mean()

        feat = torch.mm(featA, self.projs.t())
        proj_norm = F.normalize(self.projs, 2, 0)
        feat_related = torch.mm((eye_label - torch.mm(proj_norm, proj_norm.t())), featA)
        loss_sim = self.mse_loss(feat_related[sub == 0], v_feat.detach()) + self.mse_loss(feat_related[sub == 1], i_feat.detach())


        loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels, self.k_size)
        feat = self.bn_neck(feat, sub)

        logits = self.classifier(feat)
        loss_id = self.ce_loss_fn(logits.float(), labels)
        tmp = self.ce_loss_fn(logits.float(), labels)
        metric.update({'ce': tmp.data})

        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)

        logits_v = self.visible_classifier(feat[sub == 0])
        loss_id += self.ce_loss_fn(logits_v.float(), labels[sub == 0])
        logits_i = self.infrared_classifier(feat[sub == 1])
        loss_id += self.ce_loss_fn(logits_i.float(), labels[sub == 1])

        logits_m = torch.cat([logits_v, logits_i], 0).float()
        with torch.no_grad():
            self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
                                                    + self.infrared_classifier.weight.data * self.update_rate
            self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
                                                   + self.visible_classifier.weight.data * self.update_rate

            logits_v_ = self.infrared_classifier_(feat[sub == 0])
            logits_i_ = self.visible_classifier_(feat[sub == 1])
            logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()


        loss_id += self.ce_loss_fn(logits_m, logits_m_.softmax(dim=1))

        metric.update({'id': loss_id.data})
        metric.update({'cs': loss_cs.data})
        metric.update({'ceVI': loss_idVI.data})
        metric.update({'pj': loss_ortho.data})
        metric.update({'sim': loss_sim.data})
        # metric.update({'dp': loss_dp.data})

        loss = loss_id + loss_cs * self.cs_w + loss_dp * self.dp_w + loss_idVI + loss_ortho + loss_sim

        return loss, metric