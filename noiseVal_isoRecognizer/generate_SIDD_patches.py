import os
from glob import glob
from tqdm import tqdm
import numpy as np
import imageio
import random

def extractPatches(img,psize,out_name):
    h,w = img.shape[:2]
    psize = min(min(psize,h),w)
    rangex = range(0,w,psize)
    rangey = range(0,h,psize)
    ntiles = len(rangex)*len(rangey)
    counter=0
    for start_x in rangex:
        for start_y in rangey:
            end_x = start_x+psize
            end_y = start_y+psize
            if end_x > w:
                end_x = w
                start_x = end_x-psize
            if end_y > h:
                end_y = h
                start_y = end_y-psize
            tileM = img[start_y:end_y, start_x:end_x, :]
            cur_out = out_name.replace('.png','_{}.png'.format(counter))
            imageio.imsave(cur_out,tileM)
            counter=counter+1

sidd_folder = '/media/bernardo/Storage/SIDD_Small_sRGB_Only/Data'
out_dir = '/media/bernardo/Storage4T2/sidd_iso_classifier'


all_scenes = glob(sidd_folder+'/*')
counter_img=0
isos = ['00400','00800','01600','03200']
cams = ['G4', 'GP', 'IP', 'N6', 'S6']
datasets = ['L','N']
new_all_scenes = []
for i,scene in enumerate(all_scenes):
    _,_,cam,iso,_,_,dataset = os.path.basename(scene).split('_')
    if (iso in isos):
        new_all_scenes.append(scene)

all_scenes = new_all_scenes

train_slice = int(0.8*len(all_scenes))
val_slice = int(0.9*len(all_scenes))
train_scenes = all_scenes[:train_slice]
val_scenes = all_scenes[train_slice:val_slice]
test_scenes = all_scenes[val_slice:]


train_out_name = os.path.join(out_dir,'train')
for scene in tqdm(train_scenes):
    _,_,cam,iso,_,_,dataset = os.path.basename(scene).split('_')
    noisy_file = glob(scene+'/*NOISY*')[0]
    basename = os.path.basename(scene)
    noisy_img = imageio.imread(noisy_file)
    os.makedirs(os.path.join(train_out_name,iso[1:]), exist_ok=True)
    out_name = os.path.join(train_out_name,iso[1:],basename+'.png')
    extractPatches(noisy_img,psize=256,out_name=out_name)

val_out_name = os.path.join(out_dir,'val')
for scene in tqdm(val_scenes):
    _,_,cam,iso,_,_,dataset = os.path.basename(scene).split('_')
    noisy_file = glob(scene+'/*NOISY*')[0]
    basename = os.path.basename(scene)
    noisy_img = imageio.imread(noisy_file)
    os.makedirs(os.path.join(val_out_name,iso[1:]), exist_ok=True)
    out_name = os.path.join(val_out_name,iso[1:],basename+'.png')
    extractPatches(noisy_img,psize=256,out_name=out_name)


test_out_name = os.path.join(out_dir,'test')
for scene in tqdm(test_scenes):
    _,_,cam,iso,_,_,dataset = os.path.basename(scene).split('_')
    noisy_file = glob(scene+'/*NOISY*')[0]
    basename = os.path.basename(scene)
    noisy_img = imageio.imread(noisy_file)
    os.makedirs(os.path.join(test_out_name,iso[1:]), exist_ok=True)
    out_name = os.path.join(test_out_name,iso[1:],basename+'.png')
    extractPatches(noisy_img,psize=256,out_name=out_name)