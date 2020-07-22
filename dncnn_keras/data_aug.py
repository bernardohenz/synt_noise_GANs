import numpy as np


# if contrast_level is 2, the contrast will be augmented between 1/2,2    
def augment_contrast(x,y, contrast_level=2):
    tmp_rand = np.random.uniform((1/contrast_level),(contrast_level))
    outx = np.clip( (x-0.5)*tmp_rand+0.5,0,1)
    outy = np.clip( (y-0.5)*tmp_rand+0.5,0,1)
    return outx,outy
    
# if brightness_level is 0.1, the brightness will be augmented between 0.9,1.1    
def augment_brightness(x,y, brightness_level=0.2):
    tmp_rand = np.random.uniform(-brightness_level,brightness_level)
    outx = np.clip( x +tmp_rand,0,1)
    outy = np.clip( y +tmp_rand,0,1)
    return outx,outy

def random_crop(x,y, random_crop_size, sync_seed=None):
    np.random.seed(sync_seed)
    w, h = x.shape[0], x.shape[1]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew <= 0 else np.random.randint(rangew)
    offseth = 0 if rangeh <= 0 else np.random.randint(rangeh)
    return x[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1],:],y[offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1],:]

def random90rot(x,y):
   rots = np.random.randint(0,4)
   return np.rot90(x,rots,axes=(0,1)),np.rot90(y,rots,axes=(0,1))