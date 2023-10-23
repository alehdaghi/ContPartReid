import torch
import torch.nn as nn
from torch.nn import init

from part.part_model import PartModel
from resnet import resnet50, resnet18
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)




class shallow_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(shallow_module, self).__init__()

        model = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = shallow_module(arch=arch)
        self.visible_module = shallow_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])


        self.pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = DualBNNeck(self.pool_dim + 6 * 2048)
        # self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)

        # self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

        self.part_num = 7
        self.part_descriptor = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.pool_dim, 512), nn.Linear(512, 256)) for i in range(self.part_num - 1)])

        self.classifierP = nn.Linear((self.part_num - 1) * 256, class_num, bias=False)

        self.clsParts = nn.ModuleList(
            [nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, class_num)) for i in range(self.part_num - 1)])
        self.part = PartModel(self.part_num)

        self.extra_cls = nn.ModuleList(
            [nn.Linear(self.pool_dim, 2, bias=False) for i in range(9)] + [nn.Linear(self.pool_dim, 4, bias=False)]
        )

        # self.expand = nn.Sequential(
        #     nn.Conv2d(2048, 4096, kernel_size=1, padding=0, dilation=1, bias=False)
        # )
        # self.condense = nn.Conv2d(4096, 2048, kernel_size=1, padding=0, dilation=1, bias=False)

    def forward(self, x1, x2, modal=0):
        sub1 = torch.ones(x1.shape[0]) == 0
        sub2 = torch.ones(x2.shape[0]) == 1

        sub = x = torch.cat((sub1, sub2), 0)
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)

        elif modal == 1:
            x = self.visible_module(x1)
            sub = sub1
        elif modal == 2:
            x = self.thermal_module(x2)
            sub = sub2


        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            x1 = x
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            x2 = x
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            x3 = x
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet(x)
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            xx = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(xx**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        # feat_g = self.bottleneck(x_pool)
        # attr_score = [cls(feat_g) for cls in self.extra_cls]
        attr_score = [None] * 10


        part, partsFeat = self.part(x, x1, x2, x3)
        part_masks3 = F.softmax(part[0][0] + part[0][1])
        part_masks = F.softmax(F.avg_pool2d(part[0][0] + part[0][1], kernel_size=(4, 4)))
        #maskedFeat = torch.einsum('brhw, bchw -> brc', part_masks[:, 1:].detach(), x) / (h * w)
        maskedFeat2D = part_masks[:,1:].unsqueeze(2) * x.unsqueeze(1)
        maskedFeatMean = maskedFeat2D.mean(dim=[-1,-2], keepdim=True)
        maskedFeat = maskedFeatMean.squeeze(-1).squeeze(-1)
        loss_mean = (F.mse_loss(maskedFeat2D, maskedFeatMean.detach()) + F.mse_loss(maskedFeat2D.detach(), maskedFeatMean)) * 10
        maskedFeatX3 = torch.einsum('brhw, bchw -> brc', part_masks3[:, 1:], partsFeat) / (16 * h * w)
        partsScore = []
        featsP = []  # maskedFeat.sum(dim=1)
        for i in range(0, self.part_num - 1):  # 0 is background!
            # feat = self.part_descriptor[i](maskedFeat[:, i])
            partsScore.append(self.clsParts[i](maskedFeat[:, i]))
            featsP.append(maskedFeat[:, i])

        # 0: head, 1: torso, 2: upper arm, 3:lower arm, 4: upper leg, 5: lower leg
        # attr_score[0] = self.extra_cls[0](maskedFeat[:, 0]) # sex
        # attr_score[1] = self.extra_cls[1](maskedFeat[:, 0])  # long hair
        # attr_score[2] = self.extra_cls[2](maskedFeat[:, 0])  # glass
        # attr_score[3] = self.extra_cls[3](maskedFeat[:, 2] + maskedFeat[:, 3])  # long shirt
        # attr_score[4] = self.extra_cls[4](maskedFeat[:, 0] + maskedFeat[:, 1])  # V neck
        # attr_score[5] = self.extra_cls[5]( maskedFeat[:, 1])  # text on shirt
        # attr_score[6] = self.extra_cls[6](maskedFeat[:, 0] + maskedFeat[:, 1])  # is jacket
        # attr_score[7] = self.extra_cls[7](maskedFeat[:, 4] + maskedFeat[:, 5])  # is skirt
        # attr_score[8] = self.extra_cls[8](maskedFeat[:, 4] + maskedFeat[:, 5])  # is pants
        # attr_score[9] = self.extra_cls[9]( maskedFeat[:, 5])  # shoes

        featsP = torch.cat(featsP, 1)
        scoreP = None#self.classifierP(featsP)

            # feats = torch.cat([feat_g, featsP], 1)
            # attr_score = [cls(feat_g) for cls in self.extra_cls]

            # cls = part_masks.max(dim=1)
            # ids = torch.randint(1, 7, (part_masks.shape[0], 1, 1)).cuda()
            # indices = (cls == ids).unsqueeze(1).expand(-1, c, -1, -1)

            # feats = torch.cat([feat_g, featsP], 1)
        feat_b = torch.cat([x_pool, featsP], 1)
        feat = self.bottleneck(feat_b, sub)
        feat_g = feat[:, :2048]
        if self.training:
            return feat, self.classifier(feat_g), part, maskedFeatX3, maskedFeat, part_masks, partsScore, featsP, scoreP, attr_score, loss_mean
        else:
            return feat



class DualBNNeck(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.bn_neck_v = nn.BatchNorm1d(dim)
        self.bn_neck_i = nn.BatchNorm1d(dim)
        nn.init.constant_(self.bn_neck_i.bias, 0)
        nn.init.constant_(self.bn_neck_v.bias, 0)
        self.bn_neck_v.bias.requires_grad_(False)
        self.bn_neck_i.bias.requires_grad_(False)

    def forward(self, x, sub):
        mask_i = sub == 1
        mask_v = sub == 0
        bx = x.clone()
        bx[mask_i] = self.bn_neck_i(x[mask_i])
        bx[mask_v] = self.bn_neck_v(x[mask_v])

        return bx
