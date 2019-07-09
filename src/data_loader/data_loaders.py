import os
import json
import random
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # noqa

from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
import torch

from libs.torch_videovision.videotransforms import (
    video_transforms, volume_transforms, tensor_transforms
)
from base.base_data_loader import BaseDataLoader
from utils.logging_config import logger
from utils.util import channel_nb
import global_variables


HAND_KEYPOINT_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9),
    (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]


def get_hand_keypoints_color(i):
    if i == 0:
        return (255, 0, 0)
    elif 1 <= i and i <= 4:
        return (0, 255, 0)
    elif 5 <= i and i <= 8:
        return (0, 0, 255)
    elif 9 <= i and i <= 12:
        return (255, 255, 0)
    elif 13 <= i and i <= 16:
        return (0, 255, 255)
    elif 17 <= i and i <= 20:
        return (255, 0, 255)
    else:
        raise IOError(f"{i} out of range")


class KitchenDataLoader(BaseDataLoader):
    def __init__(
        self, batch_size,
        shuffle, validation_split,
        num_workers, dataset_args={},
        name='train'
    ):
        self.name = name
        self.dataset = KitchenDataSet(**dataset_args)
        super().__init__(
            self.dataset, batch_size, shuffle,
            validation_split, num_workers)

    @property
    def sample_length(self):
        return self.dataset.sample_length

    @property
    def num_classes(self):
        return len(self.dataset.verb_csv) + len(self.dataset.noun_csv)

    @property
    def num_verb_classes(self):
        return len(self.dataset.verb_csv)

    @property
    def num_noun_classes(self):
        return len(self.dataset.noun_csv)


class KitchenDataSet(Dataset):
    def __init__(
        self, rgb_root, flow_root, bbox_root, handpose_root, verb_csv, noun_csv, video_info_csv,
        test_csv=None, label_csv=None, sample_list=None,
        sample_fps=6, sample_length=32, uniform_sampling=True,
        augmentation=True, cropping_policy='none',
        prior_verb=None, prior_noun=None, pose_confidence_th=0.2,
        anticipation=False, anticipation_second=4
    ):
        self.rgb_root = rgb_root
        self.flow_root = flow_root
        self.bbox_root = bbox_root
        self.handpose_root = handpose_root
        self.sample_fps = sample_fps
        self.sample_length = sample_length
        self.uniform_sampling = uniform_sampling

        # Read csv files
        if test_csv is not None:
            self.test_mode = True
            self.augmentation = False
            self.records_csv = pd.read_csv(test_csv)
        elif label_csv is not None:
            self.test_mode = False
            self.augmentation = augmentation
            self.records_csv = pd.read_csv(label_csv)
        else:
            raise ValueError('Either test_csv or label_csv should be provided.')

        if sample_list is not None:
            with open(sample_list, 'r') as f:
                logger.info(f'Using sample list {sample_list}')
                self.sampling_idxs = [int(l) for l in f.readlines()]
        else:
            self.sampling_idxs = list(range(len(self.records_csv)))

        self.verb_csv = pd.read_csv(verb_csv)
        self.noun_csv = pd.read_csv(noun_csv)
        self.video_info_csv = pd.read_csv(video_info_csv)

        self.cropping_policy = cropping_policy
        self.modalities = global_variables.global_config.get("modalities", ["rgb"])
        self.transform_paras = {
            'rgb': {'input_mean': (0.485, 0.456, 0.406), 'input_std': (0.229, 0.224, 0.225)},
            'bbox': {'input_mean': (0, 0, 0), 'input_std': (1, 1, 1)},
            'handpose': {'input_mean': (0, 0, 0), 'input_std': (1, 1, 1)},
            'flow': {'input_mean': (0.5,), 'input_std': (0.226,)},
        }
        self.input_scale_ratios = (1, .875, .75, .66)
        self.final_image_size = (224, 224)

        self.frame_name_template = 'frame_{:010d}.jpg'
        self.flow_name_template = 'frame_{:010d}.jpg'
        self.bbox_name_template = 'entry_uid{:05d}_bbox.json'
        self.handpose_name_template = 'entry_uid{:05d}_hand_keypoints.json'
        self._load_prior(prior_verb, prior_noun)
        self.pose_confidence_th = pose_confidence_th

        self.anticipation = anticipation
        self.anticipation_second = anticipation_second

    def _load_prior(self, prior_verb, prior_noun):
        def load_prior(prior, target):
            with open(prior, 'rb') as f:
                pred = pickle.load(f)
            return {
                uid: logits for uid, logits in zip(pred['uid'], pred[target])
            }

        for prior, name in zip([prior_verb, prior_noun], ['verb', 'noun']):
            if prior is None:
                setattr(self, f'prior_{name}', None)
            else:
                if os.path.exists(prior):
                    setattr(self, f'prior_{name}', load_prior(prior, f'{name}_logits'))
                else:
                    logger.warning(f'File {prior} not exist! It will use ground truth as a prior of {name}.\n'
                                   f'Are you sure? (it only matters when conditional training is enabled)')
                    response = input('[Y/n] ')
                    if response.lower() == 'n':
                        exit()

    def _get_common_transform_list(self):
        """ Get common transform for all modalities and set random.

        All modalities will be applied with the same fixed random operation.
        """
        transform_list = [video_transforms.ResizeShorterSide(size=256)]
        if self.augmentation:
            transform_list += [
                video_transforms.GroupMultiScaleCrop(
                    input_size=self.final_image_size, scales=self.input_scale_ratios,
                    manually_set_random=True
                ),
                video_transforms.RandomHorizontalFlip(manually_set_random=True)
            ]
        else:
            if self.cropping_policy == 'none':
                pass  # It's a no-op
            elif self.cropping_policy == 'center':
                transform_list += [video_transforms.CenterCrop(size=256)]
            elif self.cropping_policy == 'multi_cropping':
                # TODO: Use something like GroupOverSampling (in the official TSN codes) to get
                #       multiple crops and average all crops' score at inference time.
                # NOTE: Temporally multiple cropping should also be implemented and tested.
                raise NotImplementedError('multi_cropping policy is not yet implemented.')
            transform_list += [video_transforms.Resize(self.final_image_size)]

        return transform_list

    def _get_transforms(self, modality, common_transform_list):
        """ Get transforms accroding to modality.

        Note that for different modalities, the common_transform_list should be the same. """
        transform_list = common_transform_list[:]  # Shallow copy, same object

        for i, transform in enumerate(transform_list):
            if type(transform) == video_transforms.RandomHorizontalFlip:
                transform_list[i].set_flow(modality == 'flow')

        # images to tensor, color normalization
        transform_list += [
            volume_transforms.ClipToTensor(channel_nb=channel_nb[modality]),
            tensor_transforms.GroupNormalize(
                self.transform_paras[modality]['input_mean'],
                self.transform_paras[modality]['input_std'])
        ]
        return video_transforms.Compose(transform_list)

    def _get_frames_idx(self, entry):
        """ Get sampled indices.

        Each entry is a video segmentation with participant_id, video_id, from start_frame to stop_frame.
        According to the sampling method given in the config, this function sample some indices
        between from start_frame to stop_frame.
        """
        participant_id, video_id = entry['participant_id'], entry['video_id']

        video_info = self.video_info_csv.loc[self.video_info_csv['video'] == video_id].iloc[0]
        fps = int(round(video_info['fps']))
        sample_stride = fps // self.sample_fps

        if self.anticipation:
            start_frame = max(0, int(entry['start_frame']) - fps * (self.anticipation_second + 1))
            end_frame = max(fps, int(entry['start_frame']) - fps)
        else:
            start_frame, end_frame = int(entry['start_frame']), int(entry['stop_frame'])
        action_length = end_frame - start_frame
        required_num = sample_stride * self.sample_length

        if (not self.uniform_sampling) and action_length > required_num:
            start = random.randrange(0, action_length - required_num)
            idxs = [i for i in range(start, start + required_num, sample_stride)]
        else:
            if not self.uniform_sampling:
                # Video length too short
                logger.debug(f'Action in {participant_id}/{video_id} ({fps} fps) too short ({action_length})'
                             f' for sample length {self.sample_length} (sample fps {self.sample_fps})')
            # all frames are sampled (some may be sampled more than once)
            rate = action_length / self.sample_length
            idxs = [int(i * rate) for i in range(self.sample_length)]

        idxs = np.array(idxs) + start_frame + 1
        return idxs

    def _get_rgb_frames(self, entry, idxs):
        participant_id, video_id = entry['participant_id'], entry['video_id']
        video_dir = os.path.join(self.rgb_root, participant_id, video_id)
        frame_images = [
            Image.open(os.path.join(video_dir, self.frame_name_template.format(i)))
            for i in idxs
        ]
        return frame_images

    def _get_bbox_frames(self, entry, idxs, size=(456, 256)):
        """ Get bounding boxes and draw them on empty frames. """
        video_dir = os.path.join(self.bbox_root)
        frame_bboxes = []
        with open(os.path.join(video_dir, self.bbox_name_template.format(entry['uid']))) as fin:
            entry_detection_data = json.load(fin)['detection_data']
        for i in idxs:
            image = Image.new("L", size)
            draw = ImageDraw.Draw(image)
            for bbox in entry_detection_data[i - idxs[0]]:
                if bbox['cls_pred'] == 0:  # Human only
                    draw.rectangle((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']), fill=255)
            frame_bboxes.append(image)
        return frame_bboxes

    def _get_flow_frames(self, entry, idxs):
        participant_id, video_id = entry['participant_id'], entry['video_id']
        video_dir = os.path.join(self.flow_root, participant_id, video_id)
        frame_flows = []
        for i in idxs:
            i = max(i, 4)
            u_image = Image.open(os.path.join(video_dir, 'u', self.flow_name_template.format((i - 2) // 2)))
            v_image = Image.open(os.path.join(video_dir, 'v', self.flow_name_template.format((i - 2) // 2)))
            # NOTE: please follow the (u, v, u, v, ...) order since RandomHorizontalFlip depends
            #       on this order to determine where to invert the flow pixel values
            frame_flows.append(u_image)
            frame_flows.append(v_image)
        return frame_flows

    def _get_handpose_frames(self, entry, idxs, size=(456, 256)):
        video_dir = os.path.join(self.handpose_root)
        frame_handposes = []
        with open(os.path.join(video_dir, self.handpose_name_template.format(entry['uid']))) as fin:
            entry_detection_data = json.load(fin)['handpose']
        for idx in idxs:
            image = Image.new("RGB", size)
            draw = ImageDraw.Draw(image)
            for person in entry_detection_data[idx - idxs[0]]:
                for hand in person:
                    for i, keypoint in enumerate(hand):
                        x = int(keypoint[0])
                        y = int(keypoint[1])
                        confidence = float(keypoint[2])
                        if confidence > self.pose_confidence_th:
                            r = 2
                            draw.ellipse((x - r, y - r, x + r, y + r), fill=get_hand_keypoints_color(i))
                    for (i, j) in HAND_KEYPOINT_PAIRS:
                        if hand[i][2] > self.pose_confidence_th and hand[j][2] > self.pose_confidence_th:
                            x1, y1 = int(hand[i][0]), int(hand[i][1])
                            x2, y2 = int(hand[j][0]), int(hand[j][1])
                            draw.line((x1, y1, x2, y2), fill=get_hand_keypoints_color(j), width=2)

            frame_handposes.append(image)
        return frame_handposes

    def _get_video_tensor(self, idxs, record_entry, modality, common_transform_list):
        transforms = self._get_transforms(modality, common_transform_list)
        if modality == 'flow':
            flow_images = self._get_flow_frames(record_entry, idxs)
            video_tensor = transforms(flow_images)
            # NOTE: Maybe it could be optimized by tensor.view()
            video_tensor = torch.cat((video_tensor[:, ::2], video_tensor[:, 1::2]), dim=0)
        elif modality == 'rgb':
            rgb_images = self._get_rgb_frames(record_entry, idxs)
            video_tensor = transforms(rgb_images)
        elif modality == 'bbox':
            bbox_images = self._get_bbox_frames(record_entry, idxs)
            video_tensor = transforms(bbox_images)
        elif modality == 'handpose':
            bbox_images = self._get_handpose_frames(record_entry, idxs)
            video_tensor = transforms(bbox_images)
        else:
            raise NotImplementedError(f"Modality {modality} not handled.")
        return video_tensor

    def __getitem__(self, index):
        # Read entry for the CSV file
        record_entry = self.records_csv.iloc[self.sampling_idxs[index]]
        uid = record_entry['uid']
        data_input = {"uid": uid}

        # Set up common_transform_list for all modalities
        common_transform_list = self._get_common_transform_list()
        idxs = self._get_frames_idx(record_entry)

        # Get each modality
        for modality in self.modalities:
            video_tensor = self._get_video_tensor(idxs, record_entry, modality, common_transform_list)
            data_input.update({f"{modality}_video": video_tensor})

            temporal_length = int(video_tensor.shape[1])
            if temporal_length != self.sample_length:
                logger.warning(f'The temporal length of loaded video tensors differs from sample_length!')

        # Load ground truth for training
        if not self.test_mode:
            data_input.update({
                "verb_class": record_entry['verb_class'],
                "noun_class": record_entry['noun_class'],
            })

        # Load prior
        if self.prior_verb is not None:
            data_input.update({'prior_verb': self.prior_verb[uid]})
        if self.prior_noun is not None:
            data_input.update({'prior_noun': self.prior_noun[uid]})

        return data_input

    def __len__(self):
        return len(self.sampling_idxs)
