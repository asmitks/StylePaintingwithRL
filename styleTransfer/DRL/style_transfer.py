import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import numpy as np

model = models.vgg19(pretrained = True).features
layers = ['2', '4', '7']

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.chosen_features = layers
    self.model = models.vgg19(pretrained=True).features

  def forward(self, x):
    features = []
    for layer_num, layer in enumerate(self.model):
      x = layer(x)
      if str(layer_num) in self.chosen_features:
        features.append(x)
    return features
model = VGG().to(device).eval()

    
def styleLoss(generated_img,style_features):
    style_loss = original_loss = 0
    generated_features = model(generated_img)
    
    for generated_feature, style_feature in zip(generated_features, style_features):
        batch_size, channel, height, width = generated_feature.shape

        # compute gram matrices
        # batch size is one so shape is batch*height*width = height*width
        # Here we are multiplying every pixel value of each channel with every other channel for the generated featuers and we will end up having shape channel by channel.
        G = generated_feature.view(channel, height*width).mm(
            generated_feature.view(channel, height*width).t()
        )

        S = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
        )

        style_loss += torch.mean((G - S)**2)
    return style_loss