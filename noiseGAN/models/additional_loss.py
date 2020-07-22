import torch
from torch.autograd import Variable

def std_for_channel(img):
   return img.contiguous().view(img.size(1), -1).std(-1)
