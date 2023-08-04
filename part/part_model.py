import einops
import torch
import torch.nn as nn
from torch.nn import init

import torchvision
import copy
import torch.nn.functional as F

from part.modules.bn import InPlaceABNSync
from part.part_detector import PSPModule, Edge_Module, Decoder_Module


class PartModel(nn.Module):
    def __init__(self,  num_part):
        super(PartModel, self).__init__()
        self.context_encoding = PSPModule(2048, 512)

        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_part)

        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_part, kernel_size=1, padding=0, dilation=1, bias=True)
        )

    def forward(self, x, x1, x2, x3):
        x = self.context_encoding(x)
        parsing_result, parsing_fea = self.decoder(x, x1)
        # Edge Branch
        edge_result, edge_fea = self.edge(x1, x2, x3)
        # Fusion Branch
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return [[parsing_result, fusion_result], [edge_result]], x
