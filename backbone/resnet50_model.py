import torch.nn as nn
import torchvision.models as models

class Resnet50Classifier(nn.module):
    def __init__(self, num_classes, pretrained=True):
        super(Resnet50Classifier, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
