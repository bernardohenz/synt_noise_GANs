import numpy as np

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh, :]

def add_uniform_noise(x, noise_intensity=0.1, **kwargs):
    out = np.clip(x+np.random.uniform(-noise_intensity,noise_intensity,size=x.shape),0,1)
    return out

def random90rot(x, **kwargs):
   rots = np.random.randint(0,3)
   out = np.rot90(x,rots,axes=(0,1))
   return out
    
    
# if contrast_level is 2, the contrast will be augmented between 1/2,2    
def augment_contrast(x, contrast_level=2, **kwargs):
    out = np.clip( (x-0.5)*np.random.uniform((1/contrast_level),(contrast_level))+0.5,0,1)
    return out
    
# if brightness_level is 0.1, the brightness will be augmented between 0.9,1.1    
def augment_brightness(x, brightness_level=0.2, **kwargs):
    out = np.clip( x + np.random.uniform(-brightness_level,brightness_level),0,1)
    return out

def check_ranges(x,**kwargs):
   if (x.max()>1):
      x = x/255.0
   return x
    
def export_img(x, folder_output, **kwargs):
    toimage(x[:, :, :], cmin=0, cmax=1).save(
        folder_output+'/'+str(random.randint(0, 99999999))+'.png')
    return x

def srgb_to_linear(srgb,**kwargs):
	linear = np.float32(srgb)# / 255.0    #assuming [0,1] pixel range
	less = linear <= 0.04045
	linear[less] = linear[less] / 12.92
	linear[~less] = np.power((linear[~less] + 0.055) / 1.055, 2.4)
	return linear
   
def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[0], x.shape[1]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew <= 0 else np.random.randint(rangew)
    offseth = 0 if rangeh <= 0 else np.random.randint(rangeh)
    return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1],:]

def compute_fft2(x, **kwargs):
   out = np.zeros_like(x)
   for i in range(x.shape[-1]):
      out[:,:,i] = np.log(np.absolute(np.fft.fftshift(np.fft.fft2(x[:,:,i])))+np.finfo(float).eps)
   out = out/np.max(out)
   return np.concatenate((x,out),axis=-1)