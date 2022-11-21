import warnings
from typing import List

import torch
import torch.nn.functional as F  # noqa
from easydict import EasyDict as edict


def binary_classification_entropy(p: torch.Tensor) -> torch.Tensor:
    """
    p: BCHW, the feature map after sigmoid, range in (0,1)
    F.bce(x,y) = -(y * logx + (1-y) * log(1-x))
    """
    # return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return F.binary_cross_entropy(p, p, reduction='none')


def multiple_classification_entropy(p: torch.Tensor, activation: str) -> torch.Tensor:
    """
    p: BCHW

    yolov5: sigmoid
    nanodet: sigmoid
    """
    assert activation in ['sigmoid', 'softmax'], f'classification type = {activation}, not in sigmoid, softmax'

    if activation == 'sigmoid':
        entropy = F.binary_cross_entropy(p, p, reduction='none')
        sum_entropy = torch.sum(entropy, dim=1, keepdim=True)
        return sum_entropy
    else:
        # for origin aldd code, use tf.log(p + 1e-12)
        entropy = -(p) * torch.log(p + 1e-7)
        sum_entropy = torch.sum(entropy, dim=1, keepdim=True)
        return sum_entropy


class FeatureMapBasedMining(object):

    def __init__(self, ymir_cfg: edict):
        self.ymir_cfg = ymir_cfg

    def mining(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        raise Exception('not implement')


class ALDDMining(FeatureMapBasedMining):
    """
    Active Learning for Deep Detection Neural Networks (ICCV 2019)
    official code: https://gitlab.com/haghdam/deep_active_learning

    change from tensorflow code to pytorch code
    1. average pooling changed, pad or not? symmetrical pad or not?
    2. max pooling changed, ceil or not?
    3. the resize shape for aggregate feature map

    those small change cause 20%-40% difference for P@N, N=100 for total 1000 images.
    P@5: 0.2
    P@10: 0.3
    P@20: 0.35
    P@50: 0.5
    P@100: 0.59
    P@200: 0.73
    P@500: 0.848
    """

    def __init__(self, ymir_cfg: edict, resize_shape: List[int]):
        super().__init__(ymir_cfg)
        self.resize_shape = resize_shape
        self.max_pool_size = 32
        self.avg_pool_size = 9
        self.align_corners = False
        self.num_classes = len(ymir_cfg.param.class_names)

    def extract_conf(self, feature_maps: List[torch.Tensor], format='yolov5') -> List[torch.Tensor]:
        """
        extract confidence feature map before sigmoid.
        """
        if format == 'yolov5':
            # feature_maps: [bs, 3, height, width, xywh + conf + num_classes]
            return [f[:, :, :, :, 4] for f in feature_maps]
        else:
            warnings.warn(f'unknown feature map format {format}')

        return feature_maps

    def mining(self, feature_maps: List[torch.Tensor]) -> torch.Tensor:
        """mining for feature maps
        feature_maps: [BCHW]
        1. resizing followed by sigmoid
        2. get mining score
        """
        # fmap = [Batch size, anchor number = 3, height, width, 5 + class_number]

        list_tmp = []
        for fmap in feature_maps:
            resized_fmap = F.interpolate(fmap, self.resize_shape, mode='bilinear', align_corners=self.align_corners)
            list_tmp.append(resized_fmap)
        conf = torch.cat(list_tmp, dim=1).sigmoid()
        scores = self.get_mining_score(conf)
        return scores

    def get_mining_score(self, confidence_feature_map: torch.Tensor) -> torch.Tensor:
        """
        confidence_feature_map: BCHW, value in (0, 1)
        1. A=sum(avg(entropy(fmap))) B,1,H,W
        2. B=sum(entropy(avg(fmap))) B,1,H,W
        3. C=max(B-A) B,1,h,w
        4. mean(C) B
        """
        avg_entropy = F.avg_pool2d(self.get_entropy(confidence_feature_map),
                                   kernel_size=self.avg_pool_size,
                                   stride=1,
                                   padding=0)
        sum_avg_entropy = torch.sum(avg_entropy, dim=1, keepdim=True)

        entropy_avg = self.get_entropy(
            F.avg_pool2d(confidence_feature_map, kernel_size=self.avg_pool_size, stride=1, padding=0))
        sum_entropy_avg = torch.sum(entropy_avg, dim=1, keepdim=True)

        uncertainty = sum_entropy_avg - sum_avg_entropy

        max_uncertainty = F.max_pool2d(uncertainty,
                                       kernel_size=self.max_pool_size,
                                       stride=self.max_pool_size,
                                       padding=0,
                                       ceil_mode=False)

        return torch.mean(max_uncertainty, dim=(1, 2, 3))

    def get_entropy(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            # binary cross entropy
            return binary_classification_entropy(feature_map)
        else:
            # multi-class cross entropy
            return multiple_classification_entropy(feature_map, activation='sigmoid')
