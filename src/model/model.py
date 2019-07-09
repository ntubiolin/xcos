import copy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

import torch
import torch.nn as nn

from libs.inflated_convnets_pytorch.src.i3res import I3ResNet
from libs.temporal_shift_module.ops.temporal_shift import make_temporal_shift, TemporalShift

from base.base_model import BaseModel
from torchvision.models import resnet50
from model.pytorch_i3d import InceptionI3d
from model import resnext
from model.tbn_resnet_C2D import TBN_resnet18, TBN_resnet50
from utils.logging_config import logger
from utils.util import one_hot_embedding, replace_module_prefix
from utils.util import extract_missing_and_unexpected_keys, count_in_channels
import global_variables


class TunableModel(BaseModel):
    def __init__(self, num_verb_classes, num_noun_classes, condition=''):
        super().__init__()
        self.num_verb_classes = num_verb_classes
        self.num_noun_classes = num_noun_classes
        self.num_classes = num_verb_classes + num_noun_classes
        self.condition = condition
        if condition != '':
            self.last_fc = nn.Linear(
                self.num_classes,
                self.num_noun_classes if condition == 'verb' else self.num_verb_classes
            )
        self.modalities = global_variables.global_config.get('modalities', ['rgb'])

    def finetune_mode_init(self):
        self.finetune_mode = ''
        self.set_finetune_mode('all')

    def set_finetune_mode(self, mode='all'):
        if self.finetune_mode == mode:
            return

        self.finetune_mode = mode
        if mode == 'all':
            for param in self.parameters():
                param.requires_grad = True
        elif mode == 'fc':
            assert hasattr(self, 'fc_layer')
            for param in self.parameters():
                param.requires_grad = False
            for param in self.fc_layer.parameters():
                param.requires_grad = True
        elif mode == 'first_conv':
            assert hasattr(self, 'first_conv')
            for param in self.parameters():
                param.requires_grad = False
            for param in self.first_conv.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f'finetune mode {mode} does not exist.')

    def forward(self, data_input):
        video_tensors = []
        for modality in self.modalities:
            video_tensors.append(data_input[f"{modality}_video"])
        cat_video_tensor = torch.cat(video_tensors, dim=1)

        logits = self.backbone(cat_video_tensor)
        results = {
            'verb_logits': logits[:, :self.num_verb_classes],
            'noun_logits': logits[:, self.num_verb_classes:]
        }
        if self.condition != '':
            if f'prior_{self.condition}' in data_input.keys():
                condition_vectors = data_input[f'prior_{self.condition}']
                condition_vectors = nn.functional.softmax(condition_vectors, dim=1)
            else:
                # Change noun/verb class to one-hot encoding as condition
                condition_vectors = one_hot_embedding(
                    data_input[f"{self.condition}_class"],
                    self.num_noun_classes if self.condition == 'noun' else self.num_verb_classes
                ).to(logits.get_device())

            # Concat conditional vector with logits and send to last_fc
            logits_str = 'verb_logits' if self.condition == 'noun' else 'noun_logits'
            last_input = torch.cat([results[logits_str], condition_vectors], dim=1)
            results[logits_str] = self.last_fc(last_input)

        return results

    def adapt_multiple_modality(self):
        modalities = global_variables.global_config.get("modalities", ["rgb"])
        if len(modalities) != 1:
            in_channels = count_in_channels(modalities)
            self.change_in_channels(in_channels)

    def change_in_channels(self, in_channels):
        raise NotImplementedError("Please implement it in your subclass.")


class ImageModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, video):
        frames = video.transpose(1, 2)
        b, t, c, h, w = list(frames.shape)

        packed_frames = frames.contiguous().view([b * t, c, h, w])
        cnn_output = self.model(packed_frames)

        logits = torch.mean(
            cnn_output.view([b, t, -1]), dim=1
        )
        assert logits.shape[1] == self.num_classes
        return logits


class Resnet50Backbone(ImageModel):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.model = resnet50(pretrained=pretrained, num_classes=self.num_classes)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)  # Replace original avgpool to adapt changeable input size


class SEResNeXt101_32x4d_Backbone(ImageModel):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        import pretrainedmodels
        self.model = pretrainedmodels.__dict__['resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        # Replace original avgpool to adapt changeable input size
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Replace the logit layer
        self.model.last_linear = nn.Linear(self.model.last_linear.in_features, self.num_classes)


class SEResNeXt101_32x4d(TunableModel):
    def __init__(self, num_verb_classes, num_noun_classes, condition=''):
        super().__init__(num_verb_classes, num_noun_classes, condition)
        self.backbone = SEResNeXt101_32x4d_Backbone(self.num_classes)
        self.fc_layer = self.backbone.model.last_linear
        self.finetune_mode_init()


class Resnet50TSM(TunableModel):
    def __init__(self, num_verb_classes, num_noun_classes, condition='',
                 load_backbone_path=None, load_trained_path=None, n_segment=16):
        super().__init__(num_verb_classes, num_noun_classes, condition)
        self.n_segment = n_segment
        self.backbone = Resnet50Backbone(self.num_classes, pretrained=False)
        make_temporal_shift(self.backbone.model, n_segment=self.n_segment, n_div=8)

        if load_backbone_path is not None:
            # load Kinetics-pretrained model provided by https://github.com/mit-han-lab/temporal-shift-module
            logger.info(f'Loading {load_backbone_path} into {self.__class__.__name__}')
            state = torch.load(load_backbone_path)['state_dict']
            # The pretrained state_dict has redundent prefixes "module.base_model."
            state = replace_module_prefix(state, 'module.', '')
            state = replace_module_prefix(state, 'base_model.', '')

            # Original fc layer has different naming, but we just
            # ignore it by setting strict=False because we don't need that
            our_state = self.backbone.model.state_dict()
            unexpected_keys = [key for key in state.keys() if key not in our_state.keys()]
            missing_keys = [key for key in our_state.keys() if key not in state.keys()]
            logger.info(f'Missing keys: {missing_keys}. Unexpeted keys: {unexpected_keys}. Going to ignore them.')
            self.backbone.model.load_state_dict(state, strict=False)

        if load_trained_path:
            # Load weights trained by this pytorch template
            # This could only be done after replace_logits()
            logger.info(f'Loading {load_trained_path} into {self.__class__.__name__}')
            state = torch.load(load_trained_path)['state_dict']
            state = {key: state[key] for key in state.keys()
                     if not key.startswith('backbone.model.conv1')}
            missing, unexpected = extract_missing_and_unexpected_keys(
                source_keys=state.keys(), target_keys=self.state_dict())
            logger.info(f'Missing keys: {missing}, unexpected keys: {unexpected}')
            self.load_state_dict(state, strict=False)

        self.adapt_multiple_modality()
        self.first_conv = self.backbone.model.conv1
        self.fc_layer = self.backbone.model.fc
        self.finetune_mode_init()

    def set_n_segment(self, n_segment):
        # NOTE: I don't know how to modify the TSM official codes gracefully to adapt
        # variable sample lengths ('n_segment' in their naming), so I write this function
        # to change that variable when data with different sample_length is encountered.
        if n_segment == self.n_segment:
            # no change, just return
            return

        logger.warning(f'Get data with sample_length = {n_segment}, '
                       f'different from that of this model ({self.n_segment}). '
                       f'Changing it from {self.n_segment} to {n_segment}...')
        self.n_segment = n_segment
        for module in self.backbone.model.modules():
            if isinstance(module, TemporalShift):
                module.set_n_segment(n_segment)

    def forward(self, data_input):
        # check if the input video length is the same as the "n_segment" of TSM modules
        one_of_modality = self.modalities[0]
        video = data_input[f'{one_of_modality}_video']
        b, c, t = video.shape[:3]
        self.set_n_segment(t)
        return super().forward(data_input)

    def change_in_channels(self, in_channels):
        """ Change input channel of the first convolutional layer if multiple modalities are used. """
        original_conv = self.backbone.model.conv1
        self.backbone.model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True
        )
        # NOTE: might be better to initialize weights by inflating that of original_conv
        self.first_conv = self.backbone.model.conv1


class ResnetI3D(TunableModel):
    def __init__(self, num_verb_classes, num_noun_classes, condition='', resnet_suffix='50'):
        super().__init__(num_verb_classes, num_noun_classes, condition)
        if resnet_suffix != '50':
            raise NotImplementedError(f'resnet{resnet_suffix} is not implemented yet.')
        resnet_2d = resnet50(pretrained=True, num_classes=self.num_classes)
        resnet_2d.avgpool = nn.AvgPool2d(7, stride=1)
        self.backbone = I3ResNet(copy.deepcopy(resnet_2d))
        self.fc_layer = self.backbone.fc
        self.finetune_mode_init()


class InceptionV1I3D(TunableModel):
    def __init__(self, num_verb_classes, num_noun_classes, condition='',
                 load_backbone_path=None, load_trained_path=None):
        super().__init__(num_verb_classes, num_noun_classes, condition)

        # When using modalities, we load the pretrained flow model
        # at first and change its first convolution layer afterwards
        if load_backbone_path:
            logger.info(f'Loading {load_backbone_path} into backbone of {self.__class__.__name__}')
            state_dict = torch.load(load_backbone_path)
            # Get the input channel from pretrained weight
            in_channels = state_dict['Conv3d_1a_7x7.conv3d.weight'].shape[1]
            self.backbone = InceptionI3d(in_channels=in_channels)
            self.backbone.load_state_dict(state_dict)
        else:
            self.backbone = InceptionI3d(in_channels=3)

        self.backbone.replace_logits(self.num_classes)
        self.fc_layer = self.backbone.logits

        if load_trained_path:
            # Load weights trained by this pytorch template
            # This could only be done after replace_logits()
            logger.info(f'Loading {load_trained_path} into {self.__class__.__name__}')
            state = torch.load(load_trained_path)['state_dict']
            state = {key: state[key] for key in state.keys()
                     if not key.startswith('backbone.Conv3d_1a_7x7.conv3d')}
            missing, unexpected = extract_missing_and_unexpected_keys(
                source_keys=state.keys(), target_keys=self.state_dict())
            logger.info(f'Missing keys: {missing}, unexpected keys: {unexpected}')
            self.load_state_dict(state, strict=False)

        self.adapt_multiple_modality()
        self.first_conv = self.backbone.end_points['Conv3d_1a_7x7']
        self.finetune_mode_init()

    def change_in_channels(self, in_channels):
        """ Change input channel of the first convolutional layer if multiple modalities are used. """
        original_conv = self.backbone.end_points['Conv3d_1a_7x7'].conv3d
        self.backbone.end_points['Conv3d_1a_7x7'].conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True
        )
        # NOTE: might be better to initialize weights by inflating that of original_conv
        self.first_conv = self.backbone.end_points['Conv3d_1a_7x7']


class ResNext101I3D(TunableModel):
    def __init__(self, num_verb_classes, num_noun_classes, condition='', load_path=None):
        super().__init__(num_verb_classes, num_noun_classes, condition)
        # Set up resnext101 I3D according to https://github.com/kenshohara/3D-ResNets-PyTorch
        self.backbone = resnext.resnet101(shortcut_type='B', cardinality=32)
        if load_path:
            logger.info(f'Loading {load_path} into {self.__class__.__name__}')
            state = torch.load(load_path)['state_dict']
            state = replace_module_prefix(state, 'module.', '')
            self.backbone.load_state_dict(state)

        # Replace the logit layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)

        self.fc_layer = self.backbone.fc
        self.finetune_mode_init()


class Resnet18TBN(TunableModel):
    def __init__(self, num_verb_classes, num_noun_classes, condition=''):
        super().__init__(num_verb_classes, num_noun_classes, condition)
        self.backbone = TBN_resnet18(pretrained=True, num_classes=self.num_classes)
        self.fc_layer = self.backbone.fc
        self.finetune_mode_init()


class Resnet50TBN(TunableModel):
    def __init__(self, num_verb_classes, num_noun_classes, condition=''):
        super().__init__(num_verb_classes, num_noun_classes, condition)
        self.backbone = TBN_resnet50(pretrained=True, num_classes=self.num_classes)
        self.fc_layer = self.backbone.fc
        self.finetune_mode_init()
