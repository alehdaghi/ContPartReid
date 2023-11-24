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

        # part, partsFeat = self.part(global_feat, x1, x2, x3)
        # part_masks3 = F.softmax(F.normalize(part[0][1]), dim=1)
        # maskedFeatX3 = torch.einsum('brhw, bchw -> brc', part_masks3, partsFeat) / (
        #             partsFeat.shape[-1] * partsFeat.shape[-2])

        # attn = part_masks3#F.avg_pool2d(part_masks3, kernel_size=(4, 4))
        # maskedFeat = maskedFeatX3#torch.einsum('brhw, bchw -> brc', attn, global_feat) / (h * w)

        # maskedFeat, attn = self.attn_pool(x4_part)
        attn = self.PRM(x4_part)
        maskedFeat = torch.einsum('brhw, bchw -> brc', attn, x4_part) / (h * w)

        global_feat = global_feat.mean(dim=(2, 3))

        if self.training:
            masks = attn.view(b, self.part_num, -1)
            if self.dp == "cos":
                loss_dp = torch.bmm(masks, masks.permute(0, 2, 1))
                loss_dp = torch.triu(loss_dp, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
                loss_dp += -masks.mean() + 1 
            elif self.dp == "l2":
                loss_dp = 0 
                for i in range(self.part_num):
                    for j in range(i+1, self.part_num):
                        loss_dp += ((((masks[:, i] - masks[:, j]) ** 2).sum(dim=1) /(18 * 9)) ** 0.5).sum()
                loss_dp = - loss_dp / (b * self.part_num * (self.part_num - 1) / 2)
                loss_dp *= self.dp_w

            F2 = einops.rearrange(maskedFeat, '(p k) ... -> p k ...', k=self.k_size) # k_size * p_size * num_part

            simGtP = 0
            for i in range(self.part_num):
                simGtP = simGtP +  (global_feat * maskedFeat[:, i]).sum(1).abs().mean()
            cont_part2 = sum([contrastive_loss(f) for f in F2]) / F2.shape[0] + simGtP/(self.part_num)

            loss_un = cont_part2   #+ 0.3*contrastive_loss(maskedFeatX3)
            partsScore = []
            for i in range(0, self.part_num):  # 0 is background!
                # feat = self.part_descriptor[i](maskedFeat[:, i])
                partsScore.append(self.clsParts[i](maskedFeat[:, i]))

            loss_pid = sum([self.ce_loss_fn(ps, labels) / 6 for ps in partsScore])


        if not self.training:
            part_feat = self.vit(maskedFeat)

            # feats = self.vit(torch.hstack([maskedFeat, global_feat.unsqueeze(1)]))
            feats_b = torch.cat([part_feat, global_feat], dim=1)
            feats = self.bn_neck(feats_b.clone(), sub)
            return feats, feats_b
        else:
            return self.train_forward(maskedFeat, global_feat, labels, loss_dp, sub, loss_un, loss_pid, **kwargs)

    def train_forward(self, maskedFeat, global_feat, labels, loss_dp, sub, loss_un, loss_pid, **kwargs):
        metric = {}
        epoch = kwargs.get('epoch')
        t = min(epoch // 20, self.part_num)

        ft1 = global_feat[sub == 0]
        ft2 = global_feat[sub == 1]
        lb1 = labels[sub == 0]
        lb2 = labels[sub == 1]
        cmp_loss = 0
        # for i in range(self.part_num):
        #     cmp_loss = cmp_loss + self.cmp(ft1, ft2, maskedFeat[:, i][sub == 0], maskedFeat[:, i][sub == 1], lb1)
        # cmp_loss = cmp_loss / self.part_num
        part_feat = self.vit(maskedFeat)
        feat = torch.cat([part_feat, global_feat], dim=1)
        loss_id = 0
        t = 100
        if t >= self.part_num:
            loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels, self.k_size)
        elif t == 0:
            loss_cs1, _, _ = self.cs_loss_fn(feat[sub==0].float(), labels[sub == 0], self.k_size // 2)
            loss_cs2, _, _ = self.cs_loss_fn(feat[sub==1].float(), labels[sub == 1], self.k_size // 2)
            loss_cs = loss_cs1 + loss_cs2
        else:
            visP, infP = self.step(t, maskedFeat, sub, labels)
            visG, infG = global_feat[sub == 0], global_feat[sub == 1]
            visP = self.vit(visP)
            infP = self.vit(infP)
            visFused = torch.cat([visP, visG], dim=1)
            infFused = torch.cat([infP, infG], dim=1)


            visPure, infPure = feat[sub == 0], feat[sub == 1]

            visFeat = torch.cat((visPure, visFused), dim=0)
            infFeat = torch.cat((infPure, infFused), dim=0)
            visFeat = einops.rearrange(visFeat, '(m p k) ... -> (p k m) ...', k=self.k_size // 2, m = 2)
            infFeat = einops.rearrange(infFeat, '(m p k) ... -> (p k m) ...', k=self.k_size // 2, m = 2)
            visFeat = self.bn_neck(visFeat, torch.zeros(labels.shape[0]))
            infFeat = self.bn_neck(infFeat, torch.ones(labels.shape[0]))

            loss_cs1, _, _ = self.cs_loss_fn(visFeat.float(), labels, self.k_size)
            loss_cs2, _, _ = self.cs_loss_fn(infFeat.float(), labels, self.k_size)
            loss_cs = loss_cs1 + loss_cs2
            logitsV = self.classifier(visFeat)
            logitsI = self.classifier(infFeat)
            loss_id = self.ce_loss_fn(logitsV.float(), labels) + self.ce_loss_fn(logitsI.float(), labels)

        loss_p_reid = self.cs_loss_fn(feat[:, :-2048].float(), labels, self.k_size)[0]
        F3 = einops.rearrange(part_feat, '(p k) ... -> k p ...',  k=self.k_size)


        loss_un = loss_un  #0.5 * contrastive_loss(F3, t=0.6)

        feat = self.bn_neck(feat, sub)
    
        logits = self.classifier(feat)
        loss_id = 0.5*loss_id + self.ce_loss_fn(logits.float(), labels)
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


        # loss_p_reid = self.cs_loss_fn(feat[:, :-2048].float(), labels, self.k_size)[0] \
                      # + self.ce_loss_fn(self.classifier_part(feat[:, :-2048]).float(), labels)

        loss_id += self.ce_loss_fn(logits_m, logits_m_.softmax(dim=1))

        # loss_ortho_1, loss_ortho_2 = 0, 0
        # proj = F.normalize(self.projs[-1], 2, 0)
        # feat_p = torch.mm(global_feat, proj)
        # proj_inner = torch.mm(proj.t(), proj)
        # eye_label = torch.eye(self.projs[0].shape[1], device=feat.device)
        # loss_ortho_2 = (proj_inner - eye_label).abs().sum(1).mean()
        # for i in range(0, self.part_num):
        #     proj = F.normalize(self.projs[i], 2, 0)
        #     part_p = torch.mm(maskedFeat[:, i], proj)
        #     loss_ortho_1 = loss_ortho_1 + (feat_p * part_p).abs().sum(1).mean()
        #
        #     proj_inner = torch.mm(proj.t(), proj)
        #     eye_label = torch.eye(self.projs[i].shape[1], device=feat.device)
        #     loss_ortho_2 = loss_ortho_2 + (proj_inner - eye_label).abs().sum(1).mean()

        # loss_ortho_2 = loss_ortho_2 / (self.part_num + 1)
        # loss_ortho_2 = loss_ortho_2 / (self.part_num )
        metric.update({'id': loss_id.data})
        metric.update({'cs': loss_cs.data})
        metric.update({'dp': loss_dp.data})
        metric.update({'un': loss_un.data})
        metric.update({'pi': loss_pid.data})
        metric.update({'pr': loss_p_reid.data})
        # metric.update({'cmp': cmp_loss.data})
        # metric.update({'o1': loss_ortho_1.data})
        # metric.update({'o2': loss_ortho_2.data})
        # metric.update({'t': t})



        loss = loss_id + loss_cs * self.cs_w + loss_dp * self.dp_w + 0.05*loss_un + 0.5*loss_pid + loss_p_reid #+ 0.05*cmp_loss#+ loss_ortho_1 + loss_ortho_2#+ loss_p_reid

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
