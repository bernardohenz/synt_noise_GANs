import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, gaussian_std, channel):
    _1D_window = gaussian(window_size, gaussian_std).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_downsample_kernel(window_size=2,channel=1):
    #box_filter
    _2D_window = torch.ones((channel,1,window_size,window_size))/4.0
    window = Variable(_2D_window)
    return window
    
def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    
    v1 = 2.0 * sigma12 + c2
    v2 = sigma1_sq + sigma2_sq + c2
    cs = (v1 / (v2+0.000001)).mean()
    #print('v1 ',str(v1.mean()))
    #print('v1 ',str(v2.mean()))
    ssim_map = ((((2.0 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2 + 0.0001))).mean()
    #ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean(),cs
    else:
        return ssim_map.mean(1).mean(1).mean(1),cs

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, gaussian_std=1.5, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.gaussian_std = gaussian_std
        self.channel = 1
        self.window = create_window(window_size, gaussian_std, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, self.gaussian_std, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.gaussian_std = gaussian_std
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, gaussian_std = 1.5,size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, gaussian_std, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    outssim,outcs = _ssim(img1, img2, window, window_size, channel, size_average)
    return (1-outssim)/2

    
def ms_ssim(img1, img2, window_size = 11, gaussian_std = 1.5,size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, gaussian_std, channel)
    downsample_window = create_downsample_kernel(2,channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        downsample_window = downsample_window.cuda(img1.get_device())
        
    window = window.type_as(img1)
    downsample_window = downsample_window.type_as(img1)
    
    weights = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size()[0]
    im1,im2 = img1.clone(),img2.clone()
    for i in range(levels):
        ssimtmp, cstmp = _ssim(im1, im2, window, window_size, channel, size_average)
        im1 = F.conv2d(im1,downsample_window,padding = 2//2, groups = channel)
        im2 = F.conv2d(im2,downsample_window,padding = 2//2, groups = channel)
        im1 = im1[:,:,::2,::2]
        im2 = im2[:,:,::2,::2]
        if i==0:
            mssim = ssimtmp
            mcs = cstmp
        else:
            mssim = torch.cat((mssim, ssimtmp))
            mcs = torch.cat((mcs,cstmp))
    all_mcs=1
    for i in range(levels):
        all_mcs = all_mcs*(mcs[i]**weights[i])
    return (1- (all_mcs*(mssim[levels - 1]**weights[levels - 1])))/2
