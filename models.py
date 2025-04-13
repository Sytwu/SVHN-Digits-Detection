# models.py

import torch
import torch.nn as nn

# torchvision detection models
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

# torchvision backbones & utils
from torchvision.models import (
    convnext_base,
    ConvNeXt_Base_Weights
)

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool


# Define Faster R-CNN model using ResNet50 as the backbone
class FasterRCNN_ResNet50_FPN(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()

        # Load pre-trained weights if specified
        weights = None
        if pretrained:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT

        # Initialize Faster R-CNN model with ResNet50-FPN backbone
        self.resnet50_fpn = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # Replace the classifier head with a new one for num_classes
        in_features = (
            self.resnet50_fpn.roi_heads.box_predictor.cls_score.in_features
        )

        self.resnet50_fpn.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def get_parameter_size(self):
        # Calculate and print the total number of parameters in millions
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")

    def get_optimizer(self, base_lr=3e-4, weight_decay=0.01):
        # Set different learning rates for different parts of the model
        low_lr = base_lr * 0.1
        optimizer = torch.optim.AdamW([
            {'params': self.resnet50_fpn.backbone.parameters(), 'lr': low_lr},
            {'params': self.resnet50_fpn.rpn.parameters(), 'lr': base_lr},
            {'params': self.resnet50_fpn.roi_heads.parameters(), 'lr': base_lr}
        ], lr=base_lr, weight_decay=weight_decay)
        return optimizer

    def forward(self, images, targets=None):
        # Forward pass through the model
        return self.resnet50_fpn(images, targets)

# End of class FasterRCNN_ResNet50_FPN ----------------------------------------


# Define a customized ConvNeXt model to expose intermediate features for FPN
class ModifiedConvNeXt(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features0 = original_model.features[0]
        self.features1 = original_model.features[1]
        self.features2 = original_model.features[2]
        self.features3 = original_model.features[3]
        self.features4 = original_model.features[4]
        self.features5 = original_model.features[5]
        self.features6 = original_model.features[6]
        self.features7 = original_model.features[7]

    def forward(self, x):
        # Extract intermediate feature maps for FPN
        out0 = self.features0(x)
        out1 = self.features1(out0)
        out2 = self.features2(out1)
        out3 = self.features3(out2)
        out4 = self.features4(out3)
        out5 = self.features5(out4)
        out6 = self.features6(out5)
        out7 = self.features7(out6)
        return {
            'features1': out1,
            'features3': out3,
            'features5': out5,
            'features7': out7
        }


# Define Faster R-CNN model using ConvNeXt as the backbone
class FasterRCNN_ConvNeXt_FPN(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()

        # Load ConvNeXt backbone (pretrained on ImageNet if specified)
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        original_convnext = convnext_base(weights=weights)
        modified_convnext = ModifiedConvNeXt(original_convnext)

        # Configure which intermediate layers to return from the backbone
        return_layers = {
            'features1': '0',
            'features3': '1',
            'features5': '2',
            'features7': '3'
        }

        # Build FPN-enhanced backbone using the selected feature layers
        backbone = IntermediateLayerGetter(
            modified_convnext, return_layers=return_layers
        )

        # Match actual output channels of selected layers
        in_channels_list = [128, 256, 512, 1024]
        backbone_with_fpn = BackboneWithFPN(
            backbone,
            in_channels_list=in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelMaxPool(),
            return_layers=return_layers
        )

        # Initialize Faster R-CNN with custom backbone + FPN
        self.faster_rcnn = FasterRCNN(
            backbone_with_fpn, num_classes=num_classes
        )

        # Replace ROI head classifier to match number of classes
        in_features = (
            self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        )

        self.faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )

    def get_parameter_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f}M")

    def get_optimizer(self, base_lr=3e-4, weight_decay=0.01):
        # Set different learning rates for different parts of the model
        low_lr = base_lr * 0.1
        optimizer = torch.optim.AdamW([
            {'params': self.faster_rcnn.backbone.parameters(), 'lr': low_lr},
            {'params': self.faster_rcnn.rpn.parameters(), 'lr': base_lr},
            {'params': self.faster_rcnn.roi_heads.parameters(), 'lr': base_lr}
        ], lr=base_lr, weight_decay=weight_decay)
        return optimizer

    def forward(self, images, targets=None):
        return self.faster_rcnn(images, targets)

# End of class FasterRCNN_ConvNeXt_FPN ----------------------------------------
