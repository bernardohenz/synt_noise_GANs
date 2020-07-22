import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import copy
import numpy as np
import os

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)

opt_copy = copy.deepcopy(opt)
opt_copy.dataset_mode  = 'aligned'
opt_copy.dataroot = opt.dataroot_aligned
opt_copy.resize_or_crop = 'resize_and_crop'
paired_data_loader = CreateDataLoader(opt_copy)
aligned_dataset = paired_data_loader.load_data()

dataset = data_loader.load_data()
dataset_size = len(data_loader)+len(paired_data_loader)
print('#training images = %d' % dataset_size)

#model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(1):#opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        if (data['A'].std()<0.005):
            print data['A_paths'][0]+ ' is corrupted, std = '+str( data['A'].std())
	    name_str=str(data['A_paths'][0])
	    command = 'mv '+name_str+' '+name_str.replace('trainA','trainA_zero_std')
	    os.system(command)
        if (data['B'].std()<0.005):
            print data['B_paths'][0]+' is corrupted, std = '+str(data['B'].std())
	    name_str=str(data['B_paths'][0])
	    command = 'mv '+name_str+' '+name_str.replace('trainB','trainB_zero_std')
	    os.system(command)
        
    for i, data in enumerate (aligned_dataset):
        if (data['A'].std()<0.005):
            print data['A_paths'][0]+' is corrupted, std = '+str(data['A'].std())
	    name_str=str(data['A_paths'][0])
	    command = 'mv '+name_str+' '+name_str.replace('train','train_zero_std')
	    os.system(command)
        if (data['B'].std()<0.005):
            print data['B_paths'][0]+' is corrupted, std = '+str(data['B'].std())

    epoch = epoch+1
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

