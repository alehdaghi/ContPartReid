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
    def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False, modality_attention=0, mutual_learning=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.pattern_attention = pattern_attention
        self.modality_attention = modality_attention
        self.mutual_learning = mutual_learning

        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride, modality_attention=modality_attention)
            D = 2048
        elif backbone == "resnet18":
            self.backbone = resnet18(pretrained=True, drop_last_stride=drop_last_stride, modality_attention=modality_attention)
            D = 512

        self.base_dim = D
        self.dim = 2048
        self.k_size = kwargs.get('k_size', 8)
        self.part_num = kwargs.get('num_parts', 7)
        self.dp = kwargs.get('dp', "l2")
        self.dp_w = kwargs.get('dp_w', 0.5)
        self.cs_w = kwargs.get('cs_w', 1.0)
        self.margin1 = kwargs.get('margin1', 0.01)
        self.margin2 = kwargs.get('margin2', 0.7)

        self.attn_pool = SAFL(part_num=self.part_num)
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

        self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num , num_classes, bias=False)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.cs_loss_fn = CSLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2)

        self.clsParts = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm1d(self.dim), nn.Linear(self.dim, num_classes, bias=False)) for i in range(self.part_num)])

        self.part = PartModel(self.part_num)
        self.vit = SimpleViT(token_size=self.part_num, num_classes=num_classes, dim=2048, depth=1)

        self.bn_neck_part = DualBNNeck(self.base_dim + self.dim * self.part_num)
        self.classifier_part = nn.Linear(self.dim * self.part_num, num_classes, bias=False)

        # self.projs = nn.ParameterList([])
        # proj = nn.Parameter(torch.zeros([self.base_dim, 512], dtype=torch.float32, requires_grad=True))
        # nn.init.kaiming_normal_(proj, nonlinearity="linear")
        # for i in range(self.part_num):
        #     proj_p = nn.Parameter(torch.zeros([self.dim, 512], dtype=torch.float32, requires_grad=True))
        #     nn.init.kaiming_normal_(proj_p, nonlinearity="linear")
        #     self.projs.append(proj_p)
        #
        # self.projs.append(proj)

        self.DEE = DEE_module(1024)
        self.PRM = PRM_module(2048, part_num=self.part_num)
        self.cmp = CPMLoss()


    
    def forward(self, inputs, labels=None, **kwargs):
        iter = kwargs.get('iteration')
        epoch = kwargs.get('epoch')
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        # CNN
        global_feat, x3, x2, x1 = self.backbone(inputs)

        x4_part = self.backbone.layer4(x3)
        if self.backbone.modality_attention > 0:
            x4_part = self.backbone.MAM4(x4_part)

        b, c, w, h = global_feat.shape
        global_feat = global_feat.mean(dim=(2, 3))
        if not self.training:

            # feats = self.vit(torch.hstack([maskedFeat, global_feat.unsqueeze(1)]))
            feats_b = global_feat
            feats = self.bn_neck(feats_b.clone(), sub)
            return feats, feats_b
        else:
            return self.train_forward(global_feat, labels, sub, **kwargs)

    def train_forward(self, feat, labels, sub, **kwargs):
        metric = {}
        epoch = kwargs.get('epoch')
        feat = self.bn_neck(feat, sub)
    
        logits = self.classifier(feat)
        loss_id = self.ce_loss_fn(logits.float(), labels)
        tmp = self.ce_loss_fn(logits.float(), labels)
        metric.update({'ce': tmp.data})

        loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels, self.k_size)

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
        # metric.update({'cmp': cmp_loss.data})
        # metric.update({'o1': loss_ortho_1.data})
        # metric.update({'o2': loss_ortho_2.data})
        # metric.update({'t': t})
        loss = loss_id + loss_cs * self.cs_w
        return loss, metric

    def step(self, t, feats, sub, labels):
        v = feats[sub == 0]
        i = feats[sub == 1]
        v_l = labels[sub == 0]
        i_l = labels[sub == 1]
        if t == 0:
            f1 = v
            f2 = i
        elif t < self.part_num:
            f1 = v.clone()
            f2 = i.clone()
            for j in range(v.shape[0]):
                index1 = np.random.choice(feats.shape[1], t, False)
                index2 = np.random.choice(feats.shape[1], t, False)
                f1[j][index1] = i[j][index1]
                f2[j][index2] = v[j][index2]
        else:
            f1 = feats[:feats.shape[0]]
            f2 = feats[feats.shape[0]:]
        return f1 , f2
