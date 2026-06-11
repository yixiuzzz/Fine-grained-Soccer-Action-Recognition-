import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import Action

class X3D(nn.Module):
    def __init__(self, num_classes, model_size='m', torch_pretrained=True):
        super(X3D, self).__init__()
        model_name = f'x3d_{model_size}'
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo', 
            model_name, pretrained=torch_pretrained)

        fs_size = self.backbone.blocks[5].proj.weight.size(1)

        # classifier
        self.backbone.blocks[5].pool.pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.backbone.blocks[5].proj = nn.Linear(fs_size, num_classes)

        # projector
        self.projector = nn.Linear(fs_size, 256)  # 2048 -> 256

        # motion Block
        for stage in range(1, 5):  # stage 1 ~ 4
            for i, res_block in enumerate(self.backbone.blocks[stage].res_blocks):
                in_channels = res_block.branch2.conv_a.in_channels
                res_block.branch2 = nn.Sequential(
                    Action(in_channels),  
                    res_block.branch2  
                )


    def forward(self, x, mask=None):
        for i in range(5):
            x = self.backbone.blocks[i](x)  

        features = x

        if mask is not None:
            features = features * mask 

        features = self.backbone.blocks[5].pool(features)
        features = features.view(features.size(0), -1)  # (B, 2048)

        # classifier
        logits = self.backbone.blocks[5].proj(features)  # (B, num_classes)

        # projector
        projected_features = self.projector(features)  # (B, 256)
        normalized_features = F.normalize(projected_features, p=2, dim=1)  

        return logits, normalized_features

if __name__ == "__main__":
    model = X3D(num_classes=4, model_size='m', torch_pretrained=False)
    with open('./x3d_m_structure.txt', 'w') as ftxt:
        ftxt.write(str(model))
