import torch.nn as nn
from torchvision import models
class Search_Model(nn.Module):
    def __init__(self):
        super(Search_Model, self).__init__()
        resnet18 = models.resnet18()
        self.features = nn.Sequential(*list(resnet18.children())[:-1])

    def forward(self, x):
        return self.features(x).squeeze()