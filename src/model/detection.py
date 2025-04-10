import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class TextDetectionModel(nn.Module):
    def __init__(self, num_classes=2):  # Background and text
        super(TextDetectionModel, self).__init__()
        
        # Load a pre-trained model for the backbone
        backbone = torchvision.models.resnet50(pretrained=True)
        
        # Remove the last two layers (avg pool and fc)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # FasterRCNN needs to know the number of output channels in the backbone
        backbone_out_channels = 2048
        
        # Define anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Define the RoI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Put the pieces together
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=800,
            max_size=1333
        )
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
