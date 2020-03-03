import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .networks import MnistGenerator, MnistDiscriminator

from .face_recog import Backbone_FC2Conv, Backbone, Am_softmax, Arcface
from .xcos_modules import XCosAttention, FrobeniusInnerProduct, GridCos, l2normalize
from utils.util import batch_visualize_xcos
# from utils.global_config import global_config

cosineDim1 = nn.CosineSimilarity(dim=1, eps=1e-6)


class xCosModel(BaseModel):
    def __init__(self,
                 net_depth=50, dropout_ratio=0.6, net_mode='ir_se',
                 model_to_plugin='CosFace', embedding_size=1568, class_num=9999,
                 use_softmax=True, softmax_temp=1, draw_qualitative_result=False):
        super().__init__()
        assert model_to_plugin in ['CosFace', 'ArcFace']
        self.attention = XCosAttention(use_softmax=True, softmax_t=1, chw2hwc=True)
        self.backbone = Backbone_FC2Conv(net_depth,
                                         dropout_ratio,
                                         net_mode)
        self.model_to_plugin = model_to_plugin
        if self.model_to_plugin == 'CosFace':
            self.head = Am_softmax(embedding_size=embedding_size,
                                   classnum=class_num)
        elif self.model_to_plugin == 'ArcFace':
            self.head = Arcface(embedding_size=embedding_size,
                                classnum=class_num)
        else:
            raise NotImplementedError
        self.backbone_target = Backbone(net_depth,
                                        dropout_ratio,
                                        net_mode)
        self.frobenius_inner_product = FrobeniusInnerProduct()
        self.grid_cos = GridCos()  # chw2hwc=True

        self.attention.weight_init(mean=0.0, std=0.02)
        self.backbone.weight_init(mean=0.0, std=0.02)
        self.backbone_target.weight_init(mean=0.0, std=0.02)

        self.draw_qualitative_result = draw_qualitative_result

    def forward(self, data_dict, scenario="normal"):
        model_output = {}
        if scenario == 'normal':
            img1s, img2s = data_dict['data_input']
            label1s, label2s = data_dict['targeted_id_labels']
            ###############
            # imgs = torch.cat((img1s, img2s), 0)
            # labels = torch.cat((label1s, label2s), 0)

            flatten_feat1s, grid_feat1s = self.backbone(img1s)
            flatten_feat2s, grid_feat2s = self.backbone(img2s)
            # Part1: FR
            theta1s = self.head(flatten_feat1s, label1s)
            theta2s = self.head(flatten_feat2s, label2s)
            # labels = torch.cat((label1s, label2s), 0)
            thetas = torch.cat((theta1s, theta2s), 0)
            # model_output["labels"] = labels
            model_output["thetas"] = thetas
            # loss1 = self.loss_fr(thetas, labels)

            # Part2: xCos
            attention_maps = self.attention(grid_feat1s, grid_feat2s)
            grid_cos_maps = self.grid_cos(grid_feat1s, grid_feat2s)
            x_coses = self.frobenius_inner_product(grid_cos_maps, attention_maps)
            targeted_coses = self.getCos(img1s, img2s)
            model_output["x_coses"] = x_coses
            model_output["targeted_cos"] = targeted_coses
        elif scenario == 'get_feature_and_xcos':
            img1s, img2s = data_dict['data_input']
            flatten_feat1s, grid_feat1s = self.backbone(img1s)
            flatten_feat2s, grid_feat2s = self.backbone(img2s)

            model_output["flatten_feats"] = (flatten_feat1s, flatten_feat2s)
            model_output["grid_feats"] = (grid_feat1s, grid_feat2s)

            attention_maps = self.attention(grid_feat1s, grid_feat2s)
            grid_cos_maps = self.grid_cos(grid_feat1s, grid_feat2s)
            x_coses = self.frobenius_inner_product(grid_cos_maps, attention_maps)
            model_output["x_coses"] = x_coses

        model_output["attention_maps"] = attention_maps
        model_output["grid_cos_maps"] = grid_cos_maps
        if self.draw_qualitative_result:
            img1s = img1s.cpu().numpy()
            img2s = img2s.cpu().numpy()
            grid_cos_maps = grid_cos_maps.squeeze().detach().cpu().numpy()
            attention_maps = attention_maps.squeeze().detach().cpu().numpy()
            visualizations = batch_visualize_xcos(img1s, img2s, grid_cos_maps, attention_maps)
            model_output["xcos_visualizations"] = visualizations
        return model_output

    def getCos(self, img1s, img2s):
        '''
        img1s.size: [bs * 2, c, h, w]
        feats: [bs * 2, 512]
        feat1: [bs, 512]
        cosine:(bs,)
        '''
        with torch.no_grad():
            feat1s = self.backbone_target(img1s)
            feat2s = self.backbone_target(img2s)
            # half_idx = feats.size(0) // 2
            # feat1 = feats[:half_idx]
            # feat2 = feats[half_idx:]
            feat1s = l2normalize(feat1s)
            feat2s = l2normalize(feat2s)
            cosine = cosineDim1(feat1s, feat2s)
            return cosine


class NormalFaceModel(BaseModel):
    def __init__(self,
                 net_depth=50, dropout_ratio=0.6, net_mode='ir_se',
                 model_type='CosFace', embedding_size=512, class_num=9999):
        super().__init__()
        assert model_type in ['CosFace', 'ArcFace']
        self.model_type = model_type
        if self.model_type == 'CosFace':
            self.head = Am_softmax(embedding_size=embedding_size,
                                   classnum=class_num)
        elif self.model_type == 'ArcFace':
            self.head = Arcface(embedding_size=embedding_size,
                                classnum=class_num)
        else:
            raise NotImplementedError
        self.backbone = Backbone(net_depth,
                                 dropout_ratio,
                                 net_mode)

        self.backbone.weight_init(mean=0.0, std=0.02)

    def forward(self, data_dict, scenario="normal"):
        model_output = {}
        if scenario == 'normal':
            img1s, img2s = data_dict['data_input']
            label1s, label2s = data_dict['targeted_id_labels']

            flatten_feat1s = self.backbone(img1s)
            flatten_feat2s = self.backbone(img2s)
            # Part1: FR
            theta1s = self.head(flatten_feat1s, label1s)
            theta2s = self.head(flatten_feat2s, label2s)
            thetas = torch.cat((theta1s, theta2s), 0)
            model_output["thetas"] = thetas

        elif scenario == 'get_feature_and_xcos':
            img1s, img2s = data_dict['data_input']
            flatten_feat1s = self.backbone(img1s)
            flatten_feat2s = self.backbone(img2s)

            model_output["flatten_feats"] = (flatten_feat1s, flatten_feat2s)
            targeted_coses = self.getCos(img1s, img2s)
            model_output["coses"] = targeted_coses
        return model_output

    def getCos(self, img1s, img2s):
        '''
        img1s.size: [bs * 2, c, h, w]
        feats: [bs * 2, 512]
        feat1: [bs, 512]
        cosine:(bs,)
        '''
        with torch.no_grad():
            feat1s = self.backbone(img1s)
            feat2s = self.backbone(img2s)
            # half_idx = feats.size(0) // 2
            # feat1 = feats[:half_idx]
            # feat2 = feats[half_idx:]
            feat1s = l2normalize(feat1s)
            feat2s = l2normalize(feat2s)
            cosine = cosineDim1(feat1s, feat2s)
            return cosine


class MnistModel(BaseModel):
    """
    Mnist model demo
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, data_dict):
        x = data_dict['data_input']
        c1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        c2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(c1)), 2))
        c2_flatten = c2.view(-1, 320)
        c2_activation = F.relu(self.fc1(c2_flatten))
        c2_dropout = F.dropout(c2_activation, training=self.training)
        fc_out = self.fc2(c2_dropout)
        out = F.log_softmax(fc_out, dim=1)
        return {
            "model_output": out
        }


class MnistGAN(BaseModel):
    def __init__(self, spectral_normalization=True, d=128):
        super().__init__()
        self.generator = MnistGenerator(d=d)
        self.discriminator = MnistDiscriminator(spectral_normalization=spectral_normalization, d=d)

        self.generator.weight_init(mean=0.0, std=0.02)
        self.discriminator.weight_init(mean=0.0, std=0.02)

    def forward(self, data_dict, scenario):
        x = data_dict['data_input']
        batch_size = x.size(0)

        # Generate images from random vector z. When inferencing, it's the only thing we need.
        z = torch.randn((batch_size, 100)).view(-1, 100, 1, 1).to(x.device)
        G_z = self.generator(z)
        model_output = {"G_z": G_z}
        if scenario == 'generator_only':
            return model_output

        # Feed fake images to the discriminator. When training generator, it's the last thing we need.
        D_G_z = self.discriminator(G_z).squeeze()
        model_output["D_G_z"] = D_G_z
        if scenario == 'generator':
            return model_output

        # Feed real images the discriminator. Only when training discriminator will this be needed.
        assert scenario == 'discriminator'
        D_x = self.discriminator(x).squeeze()
        model_output["D_x"] = D_x
        return model_output
