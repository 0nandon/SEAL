
from snn_vgg import SNN_VGG
import torch

model = SNN_VGG().cuda()

input = torch.randn(1, 2, 512, 512).cuda()
output = model(input)

breakpoint()