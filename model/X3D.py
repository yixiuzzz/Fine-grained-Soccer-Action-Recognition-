import torch
import torch.nn as nn

class X3D(nn.Module):

    def __init__(self, num_classes, model_size='m', torch_pretrained=True):
        super(X3D, self).__init__()
        model_name = f'x3d_{model_size}'
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 
            model_name, pretrained=torch_pretrained)

        fs_size = self.model.blocks[5].proj.weight.size(1)
        self.model.blocks[5].pool.pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.model.blocks[5].proj = nn.Linear(fs_size, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == "__main__":
    model = X3D(num_classes=4, model_size='m', torch_pretrained=False)
    with open('./x3d_m_structure.txt', 'w') as ftxt:
        ftxt.write(str(model.model))
