import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import gaussian_model
import sys
from .ssim_loss import ms_ssim,ssim
from .additional_loss import std_for_channel

class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        nb = opt.batchSize
        size = opt.fineSize
        self.use_lowfreq_loss = not opt.not_lowfreq_loss
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,noise_generator=False)
        
        self.netGaussian = gaussian_model.SimpleGaussian()
        self.netGaussian.apply(gaussian_model.weights_init_Gaussian)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionAligned = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionLowFreq = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A.forward(real_A)
        self.rec_A = self.netG_B.forward(fake_B).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_A = self.netG_B.forward(real_B)
        self.rec_B = self.netG_A.forward(fake_A).data
        self.fake_A = fake_A.data

        blur_real_A = self.netGaussian.forward(real_A)
        blur_fake_B = self.netGaussian.forward(fake_B)
        blur_real_B = self.netGaussian.forward(real_B)
        blur_fake_A = self.netGaussian.forward(fake_A)
        self.blur_real_A = blur_real_A.data
        self.blur_fake_B = blur_fake_B.data
        self.blur_real_B = blur_real_B.data
        self.blur_fake_A = blur_fake_A.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_std=self.opt.lambda_std
        lambda_low_freq = self.opt.lambda_low_freq
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A.forward(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B.forward(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]

        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A.forward(self.real_A)
        pred_fake = self.netD_A.forward(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B.forward(self.real_B)
        pred_fake = self.netD_B.forward(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)
    

        # Forward cycle loss
        rec_A = self.netG_B.forward(fake_B)
        
        # Backward cycle loss
        rec_B = self.netG_A.forward(fake_A)

        blur_real_A = self.netGaussian.forward(self.real_A)
        blur_real_B = self.netGaussian.forward(self.real_B)
        
        #loss_cycle_B = self.criterionCycle(rec_B, self.real_B) #* lambda_B
        if (self.opt.compare_noisy_versions == 'ssim'):
            loss_cycle_A = ssim(rec_A, self.real_A) * lambda_A
            loss_cycle_B = ssim(rec_B,self.real_B) * lambda_B
        elif (self.opt.compare_noisy_versions == 'std'):
            loss_cycle_A = self.criterionCycle( std_for_channel(rec_A),std_for_channel(self.real_A)) * lambda_std
            loss_cycle_B = self.criterionCycle( std_for_channel(rec_B),std_for_channel(self.real_B)) * lambda_std
        elif (self.opt.compare_noisy_versions == 'std_high'):
            blur_rec_A = self.netGaussian.forward(rec_A)
            blur_rec_B = self.netGaussian.forward(rec_B)
            loss_cycle_A = self.criterionCycle( std_for_channel(rec_A-blur_rec_A),std_for_channel(self.real_A-blur_real_A)) * lambda_std
            loss_cycle_B = self.criterionCycle( std_for_channel(rec_B-blur_rec_B),std_for_channel(self.real_B-blur_real_B)) * lambda_std
            #print('STD')
        else:
            loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A
            loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B
            #loss_cycle_B = self.criterionCycle(self.netGaussian.forward(rec_B),self.blur_real_B) * lambda_B
            
        #low-frequency-losses
        if self.use_lowfreq_loss:
            blur_fake_B = self.netGaussian.forward(fake_B)
            blur_fake_A = self.netGaussian.forward(fake_A)
            loss_lowfreq_A = self.criterionLowFreq(blur_fake_B,blur_real_A) * lambda_low_freq
            loss_lowfreq_B = self.criterionLowFreq(blur_fake_A,blur_real_B) * lambda_low_freq
        else:
            loss_lowfreq_A = 0
            loss_lowfreq_B = 0


        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_lowfreq_A + loss_lowfreq_B
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        if self.use_lowfreq_loss:
            self.blur_real_A = blur_real_A.data
            self.blur_fake_B = blur_fake_B.data
            self.blur_real_B = blur_real_B.data
            self.blur_fake_A = blur_fake_A.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        if self.use_lowfreq_loss:
            self.loss_lowfreq_A = loss_lowfreq_A.data[0]
            self.loss_lowfreq_B = loss_lowfreq_B.data[0]
        else:
            self.loss_lowfreq_A = 0
            self.loss_lowfreq_B =0

    
    def backward_G_aligned(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        loss_idt_A = 0
        loss_idt_B = 0
        self.loss_idt_A = 0
        self.loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)
        
        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)
        
        # Forward paired loss (smaller ISO)
        loss_paired_A = self.criterionAligned(fake_A,self.real_A) * lambda_A
        
        # Backward paired loss (higher ISO, SSIM)
        
        if (self.opt.compare_noisy_versions == 'ssim'):
            loss_paired_B = ssim(fakeB,self.real_B) * lambda_B
        elif (self.opt.compare_noisy_versions == 'std'):
            loss_paired_B = self.criterionCycle( std_for_channel(fakeB),std_for_channel(self.real_B)) * lambda_B
            #print('STD')
        else:
            loss_paired_B = self.criterionCycle(rec_B, self.real_B) * 0
        #loss_paired_B = ssim(self.real_B,fake_B) * lambda_B
            
        if self.use_lowfreq_loss:
            blur_real_A = self.netGaussian.forward(self.real_A)
            blur_fake_B = self.netGaussian.forward(fake_B)
            blur_real_B = self.netGaussian.forward(self.real_B)
            blur_fake_A = self.netGaussian.forward(fake_A)
            loss_lowfreq_A = self.criterionLowFreq(blur_fake_B,blur_real_A) * lambda_A
            loss_lowfreq_B = self.criterionLowFreq(blur_fake_A,blur_real_B) * lambda_B
        else:
            loss_lowfreq_A = 0
            loss_lowfreq_B =0
            
            loss_G = loss_G_A + loss_G_B + loss_paired_A + loss_paired_B + loss_idt_A + loss_idt_B + loss_lowfreq_A + loss_lowfreq_B
            loss_G.backward()

            self.fake_B = fake_B.data
            self.fake_A = fake_A.data

        if self.use_lowfreq_loss:
            self.blur_real_A = blur_real_A.data
            self.blur_fake_B = blur_fake_B.data
            self.blur_real_B = blur_real_B.data
            self.blur_fake_A = blur_fake_A.data

            self.loss_G_A = loss_G_A.data[0]
            self.loss_G_B = loss_G_B.data[0]
            self.loss_paired_A = loss_paired_A.data[0]
            self.loss_paired_B = loss_paired_B.data[0]
        if self.use_lowfreq_loss:
            self.loss_lowfreq_A = loss_lowfreq_A.data[0]
            self.loss_lowfreq_B = loss_lowfreq_B.data[0]
        else:
            self.loss_lowfreq_A = 0
            self.loss_lowfreq_B = 0


    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def optimize_parameters_aligned(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G_aligned()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        if self.use_lowfreq_loss:
            ret_errors['G_A_lowfreq'] = self.loss_lowfreq_A
            ret_errors['G_B_lowfreq'] = self.loss_lowfreq_B
            return ret_errors

    def get_current_errors_aligned(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Paired_A', self.loss_paired_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Paired_B',  self.loss_paired_B)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
    
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        if self.opt.isTrain and self.use_lowfreq_loss:
            ret_visuals['blur_real_A']=util.tensor2im(self.blur_real_A)
            ret_visuals['blur_fake_B']=util.tensor2im(self.blur_fake_B)
            ret_visuals['blur_real_B']=util.tensor2im(self.blur_real_B)
            ret_visuals['blur_fake_A']=util.tensor2im(self.blur_fake_A)

        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
