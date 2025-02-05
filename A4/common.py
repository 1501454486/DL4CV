"""
This module contains classes and functions that are common across both, one-stage
and two-stage detector implementations. You have to implement some parts here -
walk through the notebooks and you will find instructions on *when* to implement
*what* in this module.
"""
from typing import Dict, Tuple

import torch
from torch import _test_autograd_multiple_dispatch_view_copy, conv2d, nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction
from torchvision.ops import box_area

import sys
sys.path.append("E:/learning/computers/3DV/DL4CV assignment")
from A3.convolutional_networks import Conv


def hello_common():
    print("Hello from common.py!")


class DetectorBackboneWithFPN(nn.Module):
    r"""
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        
        # dummy_out_shapes[1]是输入通道数
        for level_name, feature_shape in dummy_out_shapes:
            in_channels = feature_shape[1]

            self.fpn_params[f'lateral_{level_name}'] = nn.Conv2d(
                in_channels = in_channels,      # 输入通道数(从backbone获取的特征通道数)
                out_channels = out_channels,    # 输出通道数(统一的FPN通道)
                kernel_size = 1,                # 1x1卷积核
                padding = 0,                    # 1x1卷积不需要填充
                stride = 1                      # 步长为1
            )

            self.fpn_params[f'output_{level_name}'] = nn.Conv2d(
                in_channels = out_channels,     # 输入通道数(与lateral层输出通道数相同)
                out_channels = out_channels,    # 输出通道数(保持不变)
                kernel_size = 3,                # 3x3卷积核
                padding = 1,                    # 填充以保持大小不变
                stride = 1                      # 步长为1
            )

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    # @property是python的一个装饰器，增加后可以像访问属性一样访问一个方法(不需要加括号)
    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        
        # 从p5到p3
        # 首先处理p5
        c5 = backbone_feats['c5']
        p5_lateral = self.fpn_params['lateral_c5'](c5)
        p5 = self.fpn_params['output_c5'](p5_lateral)
        fpn_feats["p5"] = p5
        p5_upsampled = F.interpolate( p5_lateral, scale_factor = 2, mode = 'nearest')

        # 处理p4
        c4 = backbone_feats['c4']
        p4_lateral = self.fpn_params['lateral_c4'](c4)
        p4_merged = p4_lateral + p5_upsampled
        p4 = self.fpn_params['output_c4'](p4_merged)
        fpn_feats['p4'] = p4
        p4_upsampled = F.interpolate( p4_merged, scale_factor = 2, mode = 'nearest')

        # 处理p3
        c3 = backbone_feats['c3']
        p3_lateral = self.fpn_params['lateral_c3'](c3)
        p3_merged = p3_lateral + p4_upsampled
        p3 = self.fpn_params['output_c3'](p3_merged)
        fpn_feats["p3"] = p3

        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats


def get_fpn_location_coords(
    shape_per_fpn_level: Dict[str, Tuple],
    strides_per_fpn_level: Dict[str, int],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Map every location in FPN feature map to a point on the image. This point
    represents the center of the receptive field of this location. We need to
    do this for having a uniform co-ordinate representation of all the locations
    across FPN levels, and GT boxes.

    Args:
        shape_per_fpn_level: Shape of the FPN feature level, dictionary of keys
            {"p3", "p4", "p5"} and feature shapes `(B, C, H, W)` as values.
        strides_per_fpn_level: Dictionary of same keys as above, each with an
            integer value giving the stride of corresponding FPN level.
            See `backbone.py` for more details.

    Returns:
        Dict[str, torch.Tensor]
            Dictionary with same keys as `shape_per_fpn_level` and values as
            tensors of shape `(H * W, 2)` giving `(xc, yc)` co-ordinates of the
            centers of receptive fields of the FPN locations, on input image.
    """

    # Set these to `(N, 2)` Tensors giving absolute location co-ordinates.
    location_coords = {
        level_name: None for level_name, _ in shape_per_fpn_level.items()
    }

    for level_name, feat_shape in shape_per_fpn_level.items():
        level_stride = strides_per_fpn_level[level_name]

        ######################################################################
        # TODO: Implement logic to get location co-ordinates below.          #
        ######################################################################
        # Replace "pass" statement with your code
        
        # 直观的循环写法
        # B, C, H, W = feat_shape
        # location_coord = torch.zeros( (H * W, 2), dtype = dtype, device = device )
        # for i in range(H):
        #     for j in range(W):
        #         location_coord[i * W + j, 0] = j
        #         location_coord[i * W + j, 1] = i
        # location_coords[level_name] = ( location_coord + 0.5 ) * level_stride


        # 向量版本

        _, _, H, W = feat_shape

        # 创建网格坐标
        shifts_x = torch.arange( W, dtype = dtype, device = device )
        shifts_y = torch.arange( H, dtype = dtype, device = device )

        # 生成网格点
        # torch.meshgrid会根据两个一维tensor生成二维网格
        # indexing = 'ij' 确保按照矩阵索引方式排列(i对应行，j对应列)

        # 对于 H=2, W=3 的例子，生成的网格是：
        # shift_x:
        # tensor([[0, 1, 2],
        #         [0, 1, 2]])

        # shift_y:
        # tensor([[0, 0, 0],
        #         [1, 1, 1]])
        shift_y, shift_x = torch.meshgrid( shifts_y, shifts_x, indexing = 'ij' )
        
        # 展平并堆叠
        location_coord = torch.stack(
            [shift_y.reshape(-1),       # x坐标
             shift_x.reshape(-1)],      # y坐标
             dim = -1                    # 沿维度1堆叠，即每行是一个坐标对
        )

        # 加上0.5并乘以stride得到中心点坐标
        location_coords[level_name] = ((location_coord + 0.5) * level_stride).view(-1, 2)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################
    return location_coords


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5):
    """
    Non-maximum suppression removes overlapping bounding boxes.

    Args:
        boxes: Tensor of shape (N, 4) giving top-left and bottom-right coordinates
            of the bounding boxes to perform NMS on.
        scores: Tensor of shpe (N, ) giving scores for each of the boxes.
        iou_threshold: Discard all overlapping boxes with IoU > iou_threshold

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """

    if (not boxes.numel()) or (not scores.numel()):
        return torch.zeros(0, dtype=torch.long)

    keep = None
    #############################################################################
    # TODO: Implement non-maximum suppression which iterates the following:     #
    #       1. Select the highest-scoring box among the remaining ones,         #
    #          which has not been chosen in this step before                    #
    #       2. Eliminate boxes with IoU > threshold                             #
    #       3. If any boxes remain, GOTO 1                                      #
    #       Your implementation should not depend on a specific device type;    #
    #       you can use the device of the input if necessary.                   #
    # HINT: You can refer to the torchvision library code:                      #
    # github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
    #############################################################################
    # Replace "pass" statement with your code
    
    # if boxes.numel() == 0:
    #     return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # # 1. 按分数降序排序
    # scores, idx = scores.sort(descending=True)
    # boxes = boxes[idx]

    # # 2. 初始化保留列表
    # keep = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)

    # # 3. 计算所有框之间的IoU
    # # 获取所有框的坐标
    # x1 = boxes[:, 0]
    # y1 = boxes[:, 1]
    # x2 = boxes[:, 2]
    # y2 = boxes[:, 3]

    # # 计算所有框的面积
    # areas = (x2 - x1) * (y2 - y1)  # [N]

    # for i in range(len(boxes)):
    #     if not keep[i]:
    #         continue
            
    #     # 计算当前框与剩余所有框的IoU
    #     # 只需要计算与得分较低的框的IoU
    #     xx1 = x1[i].clamp(min=x1[i + 1:])  # [N-i-1]
    #     yy1 = y1[i].clamp(min=y1[i + 1:])  # [N-i-1]
    #     xx2 = x2[i].clamp(max=x2[i + 1:])  # [N-i-1]
    #     yy2 = y2[i].clamp(max=y2[i + 1:])  # [N-i-1]

    #     # 计算交集面积
    #     w = (xx2 - xx1).clamp(min=0)  # [N-i-1]
    #     h = (yy2 - yy1).clamp(min=0)  # [N-i-1]
    #     inter = w * h  # [N-i-1]

    #     # 计算IoU
    #     ovr = inter / (areas[i] + areas[i + 1:] - inter)  # [N-i-1]
        
    #     # 将IoU大于阈值的框标记为不保留
    #     keep[i + 1:][ovr > iou_threshold] = False

    # # 返回保留的框的索引（按分数降序排序）
    # keep = idx[keep]

    if boxes.numel() == 0:
      return keep

    keep = [] # a list, convert to python long at last
    x1, y1, x2, y2 = boxes[:, :4].unbind(dim=1)
    area = torch.mul(x2 - x1, y2 - y1) # area of each boxes, shape: (N, )
    _, index = scores.sort(0) # sort the score in ascending order

    count = 0
    while index.numel() > 0:
      # keep the highest-scoring box and remove that from the index list
      largest_idx = index[-1]
      keep.append(largest_idx)
      count += 1
      index = index[:-1]
      
      # if no more box remaining, break
      if index.size(0) == 0:
        break

      # get the x1,y1,x2,y2 of all the remaining boxes, and clamp them so that
      # we get the coord of intersection of boxes and highest-scoring box
      x1_inter = torch.index_select(x1, 0, index).clamp(min=x1[largest_idx])
      y1_inter = torch.index_select(y1, 0, index).clamp(min=y1[largest_idx])
      x2_inter = torch.index_select(x2, 0, index).clamp(max=x2[largest_idx])
      y2_inter = torch.index_select(y2, 0, index).clamp(max=y2[largest_idx])

      # clamp the width and height, get the intersect area
      W_inter = (x2_inter - x1_inter).clamp(min=0.0)
      H_inter = (y2_inter - y1_inter).clamp(min=0.0)
      inter_area = W_inter * H_inter

      # retrieve the areas of all the remaining boxes, and get the union area 
      areas = torch.index_select(area, 0, index)
      union_area = (areas - inter_area) + area[largest_idx]

      # keep the boxes that have IoU <= iou_threshold
      IoU = inter_area / union_area
      index = index[IoU.le(iou_threshold)]

    # convert list to torch.long
    keep = torch.Tensor(keep).to(device=scores.device).long()

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    return keep


def class_spec_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float = 0.5,
):
    """
    Wrap `nms` to make it class-specific. Pass class IDs as `class_ids`.
    STUDENT: This depends on your `nms` implementation.

    Returns:
        keep: torch.long tensor with the indices of the elements that have been
            kept by NMS, sorted in decreasing order of scores;
            of shape [num_kept_boxes]
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = class_ids.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
