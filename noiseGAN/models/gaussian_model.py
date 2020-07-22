import torch
import torch.nn as nn

import numpy as np
import scipy

def mse(predictions,targets):
    return np.sum(((predictions - targets) ** 2))/(predictions.shape[0]*predictions.shape[1]*predictions.shape[2])

def cpsnr(img1, img2):
    mse_tmp = mse(np.round(np.clip(img1,0,255)),np.round(np.clip(img2,0,255)))
    PIXEL_MAX = 255.0
    return 10 * np.log10(PIXEL_MAX**2 / mse_tmp)
    
def gkern(l=5, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    #l should be 6x sigma
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)

class SimpleGaussian(nn.Conv2d):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=27):
        padding = kernel_size//2
        super(SimpleGaussian, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        
def weights_init_Gaussian(m):
    gauskern = gkern(m.weight.size()[-1],2.5)
    newValues = np.zeros(m.weight.size())
    for i in range(newValues.shape[0]):
        newValues[i,i,:,:] = gauskern
    m.weight.data.copy_(torch.from_numpy(newValues))
    m.weight.requires_grad=False
