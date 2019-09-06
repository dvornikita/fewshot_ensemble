import torch.nn as nn
from models.utils import distLinear as CosineSim


class ShallowNet(nn.Module):
    def __init__(self, inp_dim, hid_dim, z_dim,
                 classifier=None, num_classes=1000):
        self.classifier = classifier
        super(ShallowNet, self).__init__()

        self.block1 = self.conv_block(inp_dim, hid_dim)
        self.block2 = self.conv_block(hid_dim, hid_dim)
        self.block3 = self.conv_block(hid_dim, hid_dim)
        self.block4 = self.conv_block(hid_dim, z_dim)
        self.avg_pool = nn.AvgPool2d()
        n_feat = 1600
        if classifier == 'linear':
            self.cls_fn = nn.Linear(n_feat, num_classes)
        elif classifier == 'cosine':
            self.cls_fn = CosineSim(n_feat, num_classes)
        else:
            self.cls_fn = lambda x: x
        self.outplanes = n_feat

    def conv_block(self, in_dim, out_dim, use_relu=True):
        elements = [nn.Conv2d(in_dim, out_dim, 3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.MaxPool2d(2)]
        if use_relu:
            elements.insert(-1, nn.ReLU())
        return nn.Sequential(*elements)

    def forward(self, img):
        x = self.embed(img)
        x = self.cls_fn(x)
        return x

    def embed(self, img):
        x = self.block1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return x
