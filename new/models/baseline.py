import copy
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.MINE import Mine, estimate_MI
from models.resnet import resnet50, resnet18

from layers import CSLoss
from layers import DualBNNeck
from layers import DualBNNeck
from layers.module.part_pooling import SAFL
from models.revgrad import RevGradF
ReverseGrad = RevGradF.apply


class Baseline(nn.Module):
    def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False,
                 modality_attention=0, mutual_learning=False, **kwargs):
        super(Baseline, self).__init__()

        # From SAAI
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

        # Ours
        self.v_backbone = copy.deepcopy(self.backbone.layer4)
        self.i_backbone = copy.deepcopy(self.backbone.layer4)
        self.vi_classifier = nn.Linear(D, 2 * num_classes, bias=False)
        self.v_neck = nn.BatchNorm1d(D)
        self.i_neck = nn.BatchNorm1d(D)
        self.mine = Mine(input_size=2 * D, hidden_size=1024)
        self.mse_loss = nn.MSELoss()

        # From SAAI
        self.base_dim = D
        self.dim = D
        self.k_size = kwargs.get('k_size', 8)
        self.part_num = 0 #kwargs.get('num_parts', 7)
        self.dp = kwargs.get('dp', "l2")
        self.dp_w = kwargs.get('dp_w', 0.5)
        self.cs_w = kwargs.get('cs_w', 1.0)
        self.margin1 = kwargs.get('margin1', 0.01)
        self.margin2 = kwargs.get('margin2', 0.7)

        # From SAAI
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

        # Ours
        self._alpha = torch.tensor(0.7, requires_grad=False)

    def forward(self, inputs, labels=None, **kwargs):
        # breakpoint()
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        # CNN
        global_feat, x3, x2, x1 = self.backbone(inputs)

        # v_feat = self.v_backbone(x3[sub == 0].detach()) #detach grad
        # i_feat = self.i_backbone(x3[sub == 1].detach()) #detach grad
        v_feat = self.v_backbone(ReverseGrad(x3[sub == 0], self._alpha)) #reverse grad
        i_feat = self.i_backbone(ReverseGrad(x3[sub == 1], self._alpha)) #reverse grad
        v_feat = v_feat.mean(dim=(2, 3))
        i_feat = i_feat.mean(dim=(2, 3))


        b, c, w, h = global_feat.shape

        # part_feat, attn = self.attn_pool(global_feat)
        global_feat = global_feat.mean(dim=(2, 3))
        feats = global_feat

        if not self.training:
            feats = self.bn_neck(feats, sub)
            feats2 = torch.zeros_like(feats, device=feats.device)
            v_feat = self.v_neck(v_feat)
            i_feat = self.i_neck(i_feat)
            feats2[sub == 0] = v_feat
            feats2[sub == 1] = i_feat
            return feats, feats2
        else:
            return self.train_forward(feats, labels, 0, sub, v_feat, i_feat, **kwargs)

    def train_forward(self, featA, labels, loss_dp, sub, v_feat, i_feat, **kwargs):
        metric = {}
        breakpoint()
        labelsVI = torch.cat([2 * labels[sub == 0], 2 * labels[sub == 1] + 1], 0)
        featVI = torch.cat([v_feat, i_feat], 0)

        feats1 = torch.cat([featA[sub == 0], featA[sub == 1]], 0)

        loss_MI = 0 #estimate_MI(feats1.detach(), featVI, self.mine)

        loss_csVI, _, _ = self.cs_loss_fn(featVI.float(), labelsVI, self.k_size // 2)
        loss_cs, _, _ = self.cs_loss_fn(featA.float(), labels, self.k_size)

        loss_ortho = F.cosine_similarity(feats1.detach(), featVI).mean()

        feat = self.bn_neck(featA, sub)
        v_feat = self.v_neck(v_feat)
        i_feat = self.i_neck(i_feat)
        featVI = torch.cat([v_feat, i_feat], 0)
        logits_vi = self.vi_classifier(featVI)

        # labelsVI[sub == 0] = 2 * labels[sub == 0]
        # labelsVI[sub == 1] = 2 * labels[sub == 1] + 1
        loss_idVI = self.ce_loss_fn(logits_vi.float(), labelsVI)

        logits = self.classifier(feat)
        loss_id = self.ce_loss_fn(logits.float(), labels)
        tmp = self.ce_loss_fn(logits.float(), labels)
        metric.update({'ce': tmp.data})

        # cam_ids = kwargs.get('cam_ids')
        # sub = (cam_ids == 3) + (cam_ids == 6)



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
        metric.update({'or': loss_ortho.data})
        # metric.update({'sim': loss_sim.data})
        # metric.update({'MI':  loss_MI.data })
        metric.update({'csVI': loss_csVI.data})

        loss = loss_id + (loss_cs + loss_csVI) * self.cs_w + loss_dp * self.dp_w + loss_idVI + loss_MI #+ loss_ortho #+ 10 * loss_sim

        return loss, metric
    
class SAAI(nn.Module):
    def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False, modality_attention=0, mutual_learning=False, **kwargs):
        super(SAAI, self).__init__()

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
        self.dim = D
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
    
    def forward(self, inputs, labels=None, **kwargs):
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        # CNN
        global_feat, _, _, _ = self.backbone(inputs)

        b, c, w, h = global_feat.shape

        part_feat, attn = self.attn_pool(global_feat)
        global_feat = global_feat.mean(dim=(2, 3))
        feats = torch.cat([part_feat, global_feat], dim=1)

        if self.training:
            masks = attn.view(b, self.part_num, w*h)
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
        if not self.training:
            feats = self.bn_neck(feats, sub)
            return feats, None
        else:
            return self.train_forward(feats, labels, loss_dp, sub, **kwargs)

    def train_forward(self, feat, labels, loss_dp, sub, **kwargs):
        metric = {}

        loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels)
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
        metric.update({'dp': loss_dp.data})

        loss = loss_id + loss_cs * self.cs_w + loss_dp * self.dp_w 

        return loss, metric


class SAAI_CPR(nn.Module):
    def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False, modality_attention=0, mutual_learning=False, **kwargs):
        super(SAAI_CPR, self).__init__()

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
        self.dim = D
        self.k_size = kwargs.get('k_size', 8)
        self.part_num = kwargs.get('num_parts', 7)
        self.dp = kwargs.get('dp', "l2")
        self.dp_w = kwargs.get('dp_w', 0.5)
        self.cs_w = kwargs.get('cs_w', 1.0)
        self.margin1 = kwargs.get('margin1', 0.01)
        self.margin2 = kwargs.get('margin2', 0.7)

        # From ContPartReid: https://github.com/alehdaghi/ContPartReid/blob/Vis_Inf/new/models/baseline.py
        self.v_backbone = copy.deepcopy(self.backbone.layer4)
        self.i_backbone = copy.deepcopy(self.backbone.layer4)
        self.vi_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, 2 * num_classes, bias=False)
        self.v_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num) # changed to SAAI numbers
        self.i_neck = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num) # changed to SAAI numbers
        self.mine = Mine(input_size=2 * D, hidden_size=1024)
        self.mse_loss = nn.MSELoss()
        self._alpha = torch.tensor(0.7, requires_grad=False)

        
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
    
    def forward(self, inputs, labels=None, **kwargs):
        #breakpoint()
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        # CNN
        global_feat, x3, x2, x1 = self.backbone(inputs)

        # From ContPartReid: https://github.com/alehdaghi/ContPartReid/blob/Vis_Inf/new/models/baseline.py
        # v_feat = self.v_backbone(x3[sub == 0].detach()) #detach grad
        # i_feat = self.i_backbone(x3[sub == 1].detach()) #detach grad
        v_feat = self.v_backbone(ReverseGrad(x3[sub == 0], self._alpha)) #reverse grad
        i_feat = self.i_backbone(ReverseGrad(x3[sub == 1], self._alpha)) #reverse grad
        

        b, c, w, h = global_feat.shape
        #print(global_feat.shape, v_feat.shape, i_feat.shape)
        part_feat, attn = self.attn_pool(global_feat)
        global_feat = global_feat.mean(dim=(2, 3))
        feats = torch.cat([part_feat, global_feat], dim=1)
        #breakpoint()
        if v_feat.shape[0] != 0:
            v_part_feat = self.attn_pool.viforward(v_feat, attn[sub==0])
            v_feat = v_feat.mean(dim=(2, 3))
            v_feats = torch.cat([v_part_feat, v_feat], dim=1)
        if i_feat.shape[0] != 0:
            i_part_feat = self.attn_pool.viforward(i_feat, attn[sub==1])
            i_feat = i_feat.mean(dim=(2, 3))
            i_feats = torch.cat([i_part_feat, i_feat], dim=1)
        
        if self.training:
            masks = attn.view(b, self.part_num, w*h)
            v_masks, i_masks = attn[sub==0].view(-1, self.part_num, w*h), attn[sub==1].view(-1, self.part_num, w*h)
            if self.dp == "cos":
                loss_dp = torch.bmm(masks, masks.permute(0, 2, 1))
                loss_dp = torch.triu(loss_dp, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
                loss_dp += -masks.mean() + 1 
                v_loss_dp = torch.bmm(v_masks, v_masks.permute(0, 2, 1))
                v_loss_dp = torch.triu(v_loss_dp, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
                v_loss_dp += -v_masks.mean() + 1 
                i_loss_dp = torch.bmm(i_masks, i_masks.permute(0, 2, 1))
                i_loss_dp = torch.triu(i_loss_dp, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
                i_loss_dp += -i_masks.mean() + 1 
                loss_dp += v_loss_dp + i_loss_dp
            elif self.dp == "l2":
                loss_dp, v_loss_dp, i_loss_dp = 0, 0, 0
                for i in range(self.part_num):
                    for j in range(i+1, self.part_num):
                        loss_dp += ((((masks[:, i] - masks[:, j]) ** 2).sum(dim=1) /(18 * 9)) ** 0.5).sum()
                        v_loss_dp += ((((v_masks[:, i] - v_masks[:, j]) ** 2).sum(dim=1) /(18 * 9)) ** 0.5).sum()
                        i_loss_dp += ((((i_masks[:, i] - i_masks[:, j]) ** 2).sum(dim=1) /(18 * 9)) ** 0.5).sum()
                loss_dp = - loss_dp / (b * self.part_num * (self.part_num - 1) / 2)
                loss_dp *= self.dp_w
                v_loss_dp = - v_loss_dp / (b * self.part_num * (self.part_num - 1) / 2)
                v_loss_dp *= self.dp_w
                i_loss_dp = - i_loss_dp / (b * self.part_num * (self.part_num - 1) / 2)
                i_loss_dp *= self.dp_w
                loss_dp += v_loss_dp + i_loss_dp
        if not self.training:
            # breakpoint()
            feats = self.bn_neck(feats, sub)
            feats2 = torch.zeros_like(feats, device=feats.device)
            
            if v_feat.shape[0] != 0: 
                v_feats = self.v_neck(v_feats)
                feats2[sub == 0] = v_feats
            if i_feat.shape[0] != 0: 
                i_feats = self.i_neck(i_feats)
                feats2[sub == 1] = i_feats
            
            return feats, feats2
        else:
            return self.train_forward(feats, labels, loss_dp, sub, v_feats, i_feats, **kwargs)

    def train_forward(self, feat, labels, loss_dp, sub, v_feat, i_feat, **kwargs):
        metric = {}
        #breakpoint()
        labelsVI = torch.cat([2 * labels[sub == 0], 2 * labels[sub == 1] + 1], 0)
        featVI = torch.cat([v_feat, i_feat], 0)
        feats1 = torch.cat([feat[sub == 0], feat[sub == 1]], 0)
        
        loss_MI = 0 #estimate_MI(feats1.detach(), featVI, self.mine)
        self.cs_loss_fn.k_size = self.k_size // 2
        loss_csVI, _, _ = self.cs_loss_fn(featVI.float(), labelsVI)
        self.cs_loss_fn.k_size = self.k_size
        loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels)
        loss_ortho = F.cosine_similarity(feats1.detach(), featVI).mean()
        
        feat = self.bn_neck(feat, sub)
        v_feat = self.v_neck(v_feat)
        i_feat = self.i_neck(i_feat)
        featVI = torch.cat([v_feat, i_feat], 0)
        
        logits_vi = self.vi_classifier(featVI)
        loss_idVI = self.ce_loss_fn(logits_vi.float(), labelsVI)
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
        metric.update({'dp': loss_dp.data})
        metric.update({'ceVI': loss_idVI.data})
        metric.update({'or': loss_ortho.data})
        # metric.update({'sim': loss_sim.data})
        # metric.update({'MI':  loss_MI.data })
        metric.update({'csVI': loss_csVI.data})

        loss = loss_id + (loss_cs + loss_csVI) * self.cs_w + loss_dp * self.dp_w + loss_idVI + loss_MI

        return loss, metric
