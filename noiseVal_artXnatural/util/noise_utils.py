from skimage.util.noise import random_noise
import numpy as np
from colour_demosaicing import mosaicing_CFA_Bayer,demosaicing_CFA_Bayer_bilinear


def generateGaussianRandomVarNoise(img, std = 0.02):
    #choosen_var = np.random.uniform(0.0005,0.01) #sigma=[0,022~0.1]
    choosen_var = std**2#gaussian_stds['800']['both_dataset']**2  #CHOSEN_STD(1)[0][0]**2 #std from 1600: 0.06585
    return random_noise(img,mode='gaussian',var=choosen_var)

def generateGaussianRandomVarNoiseChannelSpecific(img, std = 0.02):
    out = np.zeros_like(img)
    for channel in range(img.shape[-1]):
      choosen_var = std**2   #CHOSEN_STD(1)[0][0]**2 #std from 1600: 0.06585 #np.random.uniform(0.0005,0.01)
      out[:,:,channel] = random_noise(img[:,:,channel],mode='gaussian',var=choosen_var)
    out = np.clip(out,0,1)
    return out
    
def generatePoissonNoise(img,std=0):
    return random_noise(img,mode='poisson')

def generateGaussianPoissonNoise(img, std = 0.02):
    gauss_out = generateGaussianRandomVarNoiseChannelSpecific(img,std)
    poiss_out = generatePoissonNoise(gauss_out)
    poiss_out = np.clip(poiss_out,0,1)
    return poiss_out

def generateDemosaicWithGaussianPoissonNoise(img,std=0.02):
    gausspois_out = generateGaussianRandomVarNoiseChannelSpecific(img,std)
    tmp_mosaic = mosaicing_CFA_Bayer(gausspois_out)
    tmp_demosaic = demosaicing_CFA_Bayer_bilinear(tmp_mosaic)
    return np.clip(tmp_demosaic,0,1)


def generateGaussianRandomVarNoiseSTD(img,std):
    choosen_var = np.random.uniform(std-0.005,std+0.005) #sigma=[0,022~0.1]
    return random_noise(img,mode='gaussian',var=choosen_var)

def generateGaussianRandomVarNoiseChannelSpecificSTD(img,std):
    out = np.zeros_like(img)
    for channel in range(img.shape[-1]):
      choosen_var = np.random.uniform(std-0.005,std+0.005)
      out[:,:,channel] = random_noise(img[:,:,channel],mode='gaussian',var=choosen_var)
    return out
    
def generateGaussianPoissonNoiseSTD(img,std):
    gauss_out = generateGaussianRandomVarNoiseChannelSpecificSTD(img,std)
    poiss_out = generatePoissonNoise(gauss_out)
    return poiss_out

def generatePoissonNoiseSTD(img,std):
    return random_noise(img,mode='poisson')

def generateDemosaicWithGaussianPoissonNoiseSTD(img,std):
    gausspois_out = generateGaussianRandomVarNoiseChannelSpecificSTD(img,std)
    tmp_mosaic = mosaicing_CFA_Bayer(gausspois_out)
    tmp_demosaic = demosaicing_CFA_Bayer_bilinear(tmp_mosaic)
    return np.clip(tmp_demosaic,0,1)