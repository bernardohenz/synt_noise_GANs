import numpy as np
from glob import glob
import imageio
import matplotlib.pyplot as plt
import random
import os
from skimage.util.noise import random_noise
from noise_utils import generateGaussianPoissonNoise
from matplotlib.font_manager import FontProperties

def kl_div_forward(p, q):
    idx = ~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))
    p = p[idx]
    q = q[idx]
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(p * np.log(p / q))

kl_divergence = kl_div_forward


def ks_value(hist_p, hist_q):
    cum_p = np.cumsum(hist_p)
    cum_q = np.cumsum(hist_q)
    return np.max( np.abs( cum_p - cum_q))


font0 = FontProperties() 
font0.set_weight('bold')
font0.set_size(32)


datasets = ['L','N']
isos = [400,800,1600,3200]
cams = ['G4','GP','IP','N6','S6']

# Creating bins for histogram
num_bins = 50
bins = np.concatenate(([-256.0], np.arange(-50, 50, 100/num_bins), [256]), axis=0)
bins = bins-0.1

# Average variances for each ISO value
gaussian_stds = {}
gaussian_stds['400'] = {}
gaussian_stds['400']['L'] = 0.04818
gaussian_stds['400']['N'] = 0.03685
gaussian_stds['400']['both_dataset'] = 0.042003
gaussian_stds['400']['both_dataset_linear'] = 0.01971
gaussian_stds['800'] = {}
gaussian_stds['800']['L'] = 0.0592269
gaussian_stds['800']['N'] = 0.0457938
gaussian_stds['800']['both_dataset'] = 0.053584
gaussian_stds['800']['both_dataset_linear'] = 0.02582
gaussian_stds['1600'] = {}
gaussian_stds['1600']['L'] = 0.0906989
gaussian_stds['1600']['N'] = 0.05532882
gaussian_stds['1600']['both_dataset'] = 0.07138
gaussian_stds['1600']['both_dataset_linear'] = 0.0343
gaussian_stds['3200'] = {}
gaussian_stds['3200']['L'] = 0.139323
gaussian_stds['3200']['N'] = 0.12520
gaussian_stds['3200']['both_dataset'] = 0.13524
gaussian_stds['3200']['both_dataset_linear'] = 0.08091

img_folder = './sample_imgs_per_ISO_lighting_camera/'  #contain only sample patches
#img_folder = '<path-to-data_per_ISO_lighting_camera>' download the zip containing all patches
out_folder = './exports_per_cam/'

os.makedirs(out_folder, exist_ok=True)

export_list = ['0126_006_S6_00400_00200_4400_L_96','0135_006_IP_00400_00400_5500_N_107',
               '0027_001_G4_00800_00350_5500_L_38','0146_007_N6_00400_00400_4400_N_155',
               '0152_007_S6_01600_01600_5500_L_89','0144_007_N6_01600_01600_4400_N_132',
               '0076_004_N6_03200_00320_3200_L_27','0014_001_S6_03200_01250_3200_N_70',
               '0123_006_G4_00400_00160_3200_N_95','0055_003_N6_00800_01000_5500_N_23',
               '0029_001_IP_00800_01000_5500_N_125','0157_007_GP_01600_01600_5500_N_150',
               '0016_001_S6_03200_01600_5500_N_75','0020_001_GP_00800_00350_5500_N_35']

export_list2 = ['0168_008_N6_00400_00200_4400_L_77','0126_006_S6_00400_00200_4400_L_96',
               '0160_007_IP_00400_00400_3200_L_116','0123_006_G4_00400_00160_3200_N_73',
               '0146_007_N6_00400_00400_4400_N_155','0135_006_IP_00400_00400_5500_N_107',
               '0106_005_GP_00400_00400_4400_N_126','0027_001_G4_00800_00350_5500_L_38',
               '0075_004_N6_00800_00080_3200_L_71','0011_001_S6_00800_00500_5500_L_175',
               '0019_001_GP_00800_00640_5500_L_73','0055_003_N6_00800_01000_5500_N_23',
               '0010_001_S6_00800_00350_3200_N_59','0032_001_IP_00800_01000_3200_N_61',
               '0020_001_GP_00800_00350_5500_N_35','0120_006_N6_01600_00400_3200_L_147',
               '0152_007_S6_01600_01600_5500_L_89','0144_007_N6_01600_01600_4400_N_132',
               '0052_002_S6_01600_01000_5500_N_108','0042_002_IP_01600_03100_5500_N_90',
               '0064_003_GP_01600_01600_4400_N_37','0076_004_N6_03200_00320_3200_L_27',
               '0013_001_S6_03200_01250_3200_L_173','0145_007_N6_03200_03200_4400_N_178',
               '0014_001_S6_03200_01250_3200_N_70','0182_008_GP_03200_03200_5500_N_7']

export_list_smaller = ['0123_006_G4_00400_00160_3200_N_95','0055_003_N6_00800_01000_5500_N_38',
                       '0029_001_IP_00800_01000_5500_N_125','0157_007_GP_01600_01600_5500_N_150',
                       '0014_001_S6_03200_01250_3200_N_89']
export_list = export_list + export_list2 + export_list_smaller
print("Kl divergence and KS metrics. The lower the better.")
print('=====================')
for cur_iso in isos:
    for dataset in datasets:
        cur_dataset_num_images = 0
        cur_dataset_hist_natural = np.zeros(((bins.size-1),3))
        cur_dataset_hist_natural_NF = np.zeros(((bins.size-1),3))
        cur_dataset_hist_gauss = np.zeros(((bins.size-1),3))
        cur_dataset_hist_awgn_poi_linear = np.zeros(((bins.size-1),3))
        cur_dataset_hist_noiseflow = np.zeros(((bins.size-1),3))
        cur_dataset_hist_gan = np.zeros(((bins.size-1),3))

        for cur_cam in cams:
            cur_folder = os.path.join(img_folder,dataset,str(cur_iso),cur_cam)
            img_names = glob(cur_folder+"/samples_gts/*.png")
            num_images = len(img_names)
            if (num_images==0):
                continue
            cur_dataset_num_images += num_images

            cur_cam_ks_noiseflow = np.zeros((num_images,))
            cur_cam_kl_noiseflow = np.zeros((num_images,))
            cur_cam_ks_gan = np.zeros((num_images,))
            cur_cam_kl_gan = np.zeros((num_images,))
            cur_cam_ks_gaussian = np.zeros((num_images,))
            cur_cam_kl_gaussian = np.zeros((num_images,))
            cur_cam_hist_natural = np.zeros(((bins.size-1),3))
            cur_cam_hist_natural_NF = np.zeros(((bins.size-1),3))
            cur_cam_hist_gauss = np.zeros(((bins.size-1),3))
            cur_cam_hist_awgn_poi_linear = np.zeros(((bins.size-1),3))
            cur_cam_hist_noiseflow = np.zeros(((bins.size-1),3))
            cur_cam_hist_gan = np.zeros(((bins.size-1),3))

            for i,img_name in enumerate(img_names):
                # should_print = bool([ele for ele in export_list if(ele in img_name)] )
                # if (not should_print):
                #      continue

                img_clean = imageio.imread(img_name).astype('float32')/255.0
                img_noisy = imageio.imread(img_name.replace('samples_gts','samples_noisy').replace('GT_SRGB','NOISY_SRGB')).astype('float32')/255.0
                img_clean_NF = imageio.imread(img_name.replace('samples_gts','samples_gtsNF')).astype('float32')/255.0
                img_noisy_NF = imageio.imread(img_name.replace('samples_gts','samples_noisyNF').replace('GT_SRGB','NOISY_SRGB')).astype('float32')/255.0
                img_gan = imageio.imread(img_name.replace('samples_gts','GAN').replace('.png','_fake_B.png')).astype('float32')/255.0
                #std = np.random.uniform(0.24,11.51)/255.0  #Noiseflow values
                std = gaussian_stds[str(cur_iso)]['both_dataset']
                std_linear = gaussian_stds[str(cur_iso)]['both_dataset_linear']
                img_gaussian = random_noise(img_clean,mode='gaussian',var=std**2)
                #img_gaussian = np.clip(random_noise(img_clean**2.2,mode='gaussian',var=std_linear**2)**(1/2.2),0,1) ## AWGN linear
                #img_awgn_poi_linear = generateGaussianPoissonNoise(img_clean,std)
                img_awgn_poi_linear = np.clip(generateGaussianPoissonNoise((img_clean**2.2),std_linear)**(1/2.2),0,1) ## GaussianPoissonian in linear
                #img_gaussian = imageio.imread(img_name.replace('samples_GT_sRGB','samples_gaussian_sRGB')).astype('float32')/255.0
                img_noiseflow = imageio.imread(img_name.replace('samples_gts','samples_noiseflowNF')).astype('float32')/255.0

                #Computing diffs
                diff_natural = img_noisy-img_clean
                diff_natural_NF = img_noisy_NF - img_clean_NF
                diff_gan = img_gan-img_clean
                diff_gauss = img_gaussian-img_clean
                diff_awgn_poi_linear = img_awgn_poi_linear-img_clean
                diff_noiseflow = img_noiseflow-img_clean_NF

                #Computing histograms
                hist_natural_rgb = np.zeros(((bins.size-1),3))
                hist_gauss_rgb = np.zeros(((bins.size-1),3))
                hist_awgn_poi_linear_rgb = np.zeros(((bins.size-1),3))
                hist_gan_rgb = np.zeros(((bins.size-1),3))
                hist_noiseflow_NF = np.zeros(((bins.size-1),3))
                hist_natural_rgb_NF = np.zeros(((bins.size-1),3))
                colors=['red','green','blue']
                for channel in range(3):
                    hist_natural,_ = np.histogram(diff_natural[:,:,channel]*255,bins=bins,range=(-255,255),density=False)
                    hist_natural_NF,_ = np.histogram(diff_natural_NF[:,:,channel]*255,bins=bins,range=(-255,255),density=False)
                    hist_gauss,_ = np.histogram(diff_gauss[:,:,channel]*255,bins=bins,range=(-255,255),density=False)
                    hist_awgn_poi_linear,_ = np.histogram(diff_awgn_poi_linear[:,:,channel]*255,bins=bins,range=(-255,255),density=False)
                    hist_gan,_ = np.histogram(diff_gan[:,:,channel]*255,bins=bins,range=(-255,255),density=False)
                    hist_noiseflow,_ = np.histogram(diff_noiseflow[:,:,channel]*255,bins=bins,range=(-255,255),density=False)
                    hist_natural_rgb[:,channel] = hist_natural
                    hist_gauss_rgb[:,channel] = hist_gauss
                    hist_awgn_poi_linear_rgb[:,channel] = hist_awgn_poi_linear
                    hist_gan_rgb[:,channel] = hist_gan
                    hist_noiseflow_NF[:,channel] = hist_noiseflow
                    hist_natural_rgb_NF[:,channel] = hist_natural_NF
                hist_natural = hist_natural_rgb
                hist_natural_NF = hist_natural_rgb_NF
                hist_gauss = hist_gauss_rgb
                hist_awgn_poi_linear = hist_awgn_poi_linear_rgb
                hist_gan = hist_gan_rgb
                hist_noiseflow = hist_noiseflow_NF

                #Concatenating in histogram per cam
                cur_cam_hist_natural = cur_cam_hist_natural + hist_natural
                cur_cam_hist_natural_NF = cur_cam_hist_natural_NF + hist_natural_rgb_NF
                cur_cam_hist_gan = cur_cam_hist_gan + hist_gan
                cur_cam_hist_gauss = cur_cam_hist_gauss + hist_gauss
                cur_cam_hist_awgn_poi_linear = cur_cam_hist_awgn_poi_linear + hist_awgn_poi_linear
                cur_cam_hist_noiseflow = cur_cam_hist_noiseflow + hist_noiseflow

                # Computing KL divergence
                kl_gan = 0
                kl_gauss = 0
                kl_awgn_poi_linear = 0
                kl_noiseflow = 0
                normalize_size = img_clean.shape[0]*img_clean.shape[1]
                for channel in range(3):
                    kl_gan = kl_gan + kl_divergence(hist_natural[:,channel]/normalize_size,hist_gan[:,channel]/normalize_size)
                    kl_gauss = kl_gauss + kl_divergence(hist_natural[:,channel]/normalize_size,hist_gauss[:,channel]/normalize_size)
                    kl_awgn_poi_linear = kl_awgn_poi_linear + kl_divergence(hist_natural[:,channel]/normalize_size,hist_awgn_poi_linear[:,channel]/normalize_size)
                    kl_noiseflow = kl_noiseflow + kl_divergence(hist_natural_NF[:,channel]/normalize_size,hist_noiseflow[:,channel]/normalize_size)
                kl_gan = kl_gan/3
                kl_gauss = kl_gauss/3
                kl_awgn_poi_linear = kl_awgn_poi_linear/3
                kl_noiseflow = kl_noiseflow/3

                # Computing KS-value
                ks_gan = 0
                ks_gauss = 0
                ks_awgn_poi_linear = 0
                ks_noiseflow = 0
                for channel in range(3):
                    ks_gan = ks_gan + ks_value(hist_natural[:,channel]/normalize_size,hist_gan[:,channel]/normalize_size)
                    ks_gauss = ks_gauss + ks_value(hist_natural[:,channel]/normalize_size,hist_gauss[:,channel]/normalize_size)
                    ks_awgn_poi_linear = ks_awgn_poi_linear + ks_value(hist_natural[:,channel]/normalize_size,hist_awgn_poi_linear[:,channel]/normalize_size)
                    ks_noiseflow = ks_noiseflow + ks_value(hist_natural_NF[:,channel]/normalize_size,hist_noiseflow[:,channel]/normalize_size)

                ks_gan /= 3
                ks_gauss /= 3
                ks_awgn_poi_linear /= 3
                ks_noiseflow /=3

                # Export only X% of images
                should_print = bool([ele for ele in export_list if(ele in img_name)] )
                if (False): # Replace by if (should_print):    for exporting comparison patches
                    cur_dataset_folder = os.path.join(out_folder,dataset)
                    cur_iso_folder = os.path.join(cur_dataset_folder,str(cur_iso))
                    cur_cam_folder = os.path.join(cur_iso_folder,cur_cam)
                    os.makedirs(cur_dataset_folder, exist_ok=True)
                    os.makedirs(cur_iso_folder, exist_ok=True)
                    os.makedirs(cur_cam_folder, exist_ok=True)
                    out_img_path = os.path.join(cur_cam_folder,os.path.basename(img_name))

                    fig1, ax1 = plt.subplots()
                    ax1.imshow(img_gan)
                    ax1.text(5, 28, "KL = {0:.4f}".format(kl_gan), fontproperties=font0, fontdict=dict(color='white'), bbox=dict(fill=True, color='blue', linewidth=2))
                    ax1.text(5, 66, "KS = {0:.4f}".format(ks_gan), fontproperties=font0, fontdict=dict(color='white'), bbox=dict(fill=True, color='blue', linewidth=2))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_gan.pdf'))
                    plt.close(fig1)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(img_gaussian)
                    ax1.text(5, 28, "KL = {0:.4f}".format(kl_gauss), fontproperties=font0, fontdict=dict(color='white'), bbox=dict(fill=True, color='brown', linewidth=2))
                    ax1.text(5, 66, "KS = {0:.4f}".format(ks_gauss), fontproperties=font0, fontdict=dict(color='white'), bbox=dict(fill=True, color='brown', linewidth=2))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_gaussian.pdf'))
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(img_awgn_poi_linear)
                    ax1.text(5, 28, "KL = {0:.4f}".format(kl_awgn_poi_linear), fontproperties=font0, fontdict=dict(color='white'), bbox=dict(fill=True, color='darkmagenta', linewidth=2))
                    ax1.text(5, 66, "KS = {0:.4f}".format(ks_awgn_poi_linear), fontproperties=font0, fontdict=dict(color='white'), bbox=dict(fill=True, color='darkmagenta', linewidth=2))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_awgn_poi_linear.pdf'))
                    plt.close(fig1)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(np.clip(img_noiseflow+img_clean-img_clean_NF,0,1))
                    ax1.text(5, 28, "KL = {0:.4f}".format(kl_noiseflow), fontproperties=font0, fontdict=dict(color='white'), bbox=dict(fill=True, color='green', linewidth=2))
                    ax1.text(5, 66, "KS = {0:.4f}".format(ks_noiseflow), fontproperties=font0, fontdict=dict(color='white'), bbox=dict(fill=True, color='green', linewidth=2))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_noiseflow.pdf'))
                    plt.close(fig1)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(img_clean)
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_clean.pdf'))
                    plt.close(fig1)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(img_noisy)
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_noisy.pdf'))
                    plt.close(fig1)

                    ## Plotting residual (noisy-clean)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(np.clip(np.absolute(diff_gan)*3,0,1))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_gan_diff.pdf'))
                    plt.close(fig1)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(np.clip(np.absolute(diff_gauss)*3,0,1))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_gaussian_diff.pdf'))
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(np.clip(np.absolute(diff_awgn_poi_linear)*3,0,1))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_awgn_poi_linear_diff.pdf'))
                    plt.close(fig1)
                    fig1, ax1 = plt.subplots()
                    diff_noiseflow_for_show = np.clip(np.clip(img_noiseflow+img_clean-img_clean_NF,0,1) - img_clean,0,1)
                    ax1.imshow(np.clip(np.absolute(diff_noiseflow_for_show)*3,0,1))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_noiseflow_diff.pdf'))
                    plt.close(fig1)
                    fig1, ax1 = plt.subplots()
                    ax1.imshow(np.clip(np.absolute(diff_natural)*3,0,1))
                    ax1.axis('off')
                    fig1.savefig(out_img_path.replace('.png','_noisy_diff.pdf'))
                    plt.close(fig1)
                    
            print('Dataset {}_{}_{}'.format(cur_iso,dataset,cur_cam))
            kl_gan = 0
            kl_gauss = 0
            kl_awgn_poi_linear =0
            kl_noiseflow = 0
            normalize_size = num_images*img_clean.shape[0]*img_clean.shape[1]
            for channel in range(3):
                kl_gan = kl_gan + kl_divergence(cur_cam_hist_natural[:,channel]/normalize_size,cur_cam_hist_gan[:,channel]/normalize_size)
                kl_gauss = kl_gauss + kl_divergence(cur_cam_hist_natural[:,channel]/normalize_size,cur_cam_hist_gauss[:,channel]/normalize_size)
                kl_awgn_poi_linear = kl_awgn_poi_linear + kl_divergence(cur_cam_hist_natural[:,channel]/normalize_size,cur_cam_hist_awgn_poi_linear[:,channel]/normalize_size)
                kl_noiseflow = kl_noiseflow + kl_divergence(cur_cam_hist_natural_NF[:,channel]/normalize_size,cur_cam_hist_noiseflow[:,channel]/normalize_size)
            kl_gan = kl_gan/3
            kl_gauss = kl_gauss/3
            kl_awgn_poi_linear /= 3
            kl_noiseflow = kl_noiseflow/3

            ks_gan = 0
            ks_gauss = 0
            ks_awgn_poi_linear = 0
            ks_noiseflow = 0
            for channel in range(3):
                cum_sum_natural_hist = np.cumsum(cur_cam_hist_natural[:,channel]/normalize_size)
                ks_gan = ks_gan + np.max( np.abs( cum_sum_natural_hist - np.cumsum(cur_cam_hist_gan[:,channel]/normalize_size) ))
                ks_gauss = ks_gauss + np.max( np.abs( cum_sum_natural_hist - np.cumsum(cur_cam_hist_gauss[:,channel]/normalize_size) ))
                ks_awgn_poi_linear = ks_gauss + np.max( np.abs( cum_sum_natural_hist - np.cumsum(cur_cam_hist_awgn_poi_linear[:,channel]/normalize_size) ))
                ks_noiseflow = ks_noiseflow + np.max( np.abs( np.cumsum(cur_cam_hist_natural_NF[:,channel]/normalize_size) - np.cumsum(cur_cam_hist_noiseflow[:,channel]/normalize_size) ))
            ks_gan /= 3
            ks_gauss /= 3
            ks_awgn_poi_linear /=3
            ks_noiseflow /=3
            print('---')
            print('KL gauss: {0:.4f}'.format(kl_gauss))
            print('KL awgn_poi_linear: {0:.4f}'.format(kl_awgn_poi_linear))
            print('KL noiseflow: {0:.4f}'.format(kl_noiseflow))
            print('KL gan: {0:.4f}'.format(kl_gan))
            print('---')
            print('KS gauss: {0:.4f}'.format(ks_gauss))
            print('KS awgn_poi_linear: {0:.4f}'.format(ks_awgn_poi_linear))
            print('KS noiseflow: {0:.4f}'.format(ks_noiseflow))
            print('KS gan: {0:.4f}'.format(ks_gan))
            print('========')
            cur_dataset_hist_natural = cur_dataset_hist_natural + cur_cam_hist_natural
            cur_dataset_hist_natural_NF = cur_dataset_hist_natural_NF + cur_cam_hist_natural_NF
            cur_dataset_hist_gauss = cur_dataset_hist_gauss + cur_cam_hist_gauss
            cur_dataset_hist_awgn_poi_linear = cur_dataset_hist_awgn_poi_linear + cur_cam_hist_awgn_poi_linear
            cur_dataset_hist_noiseflow = cur_dataset_hist_noiseflow + cur_cam_hist_noiseflow
            cur_dataset_hist_gan = cur_dataset_hist_gan + cur_cam_hist_gan

        print('Dataset {}_{}'.format(cur_iso,dataset))
        kl_gan = 0
        kl_gauss = 0
        kl_awgn_poi_linear = 0
        kl_noiseflow = 0
        normalize_size = cur_dataset_num_images*img_clean.shape[0]*img_clean.shape[1]
        for channel in range(3):
            kl_gan = kl_gan + kl_divergence(cur_dataset_hist_natural[:,channel]/normalize_size,cur_dataset_hist_gan[:,channel]/normalize_size)
            kl_gauss = kl_gauss + kl_divergence(cur_dataset_hist_natural[:,channel]/normalize_size,cur_dataset_hist_gauss[:,channel]/normalize_size)
            kl_awgn_poi_linear = kl_awgn_poi_linear + kl_divergence(cur_dataset_hist_natural[:,channel]/normalize_size,cur_dataset_hist_awgn_poi_linear[:,channel]/normalize_size)
            kl_noiseflow = kl_noiseflow + kl_divergence(cur_dataset_hist_natural_NF[:,channel]/normalize_size,cur_dataset_hist_noiseflow[:,channel]/normalize_size)
        kl_gan = kl_gan/3
        kl_gauss = kl_gauss/3
        kl_awgn_poi_linear /=3
        kl_noiseflow = kl_noiseflow/3

        ks_gan = 0
        ks_gauss = 0
        ks_awgn_poi_linear = 0
        ks_noiseflow = 0
        for channel in range(3):
            cum_sum_natural_hist = np.cumsum(cur_dataset_hist_natural[:,channel]/normalize_size)
            ks_gan = ks_gan + np.max( np.abs( cum_sum_natural_hist - np.cumsum(cur_dataset_hist_gan[:,channel]/normalize_size) ))
            ks_gauss = ks_gauss + np.max( np.abs( cum_sum_natural_hist - np.cumsum(cur_dataset_hist_gauss[:,channel]/normalize_size) ))
            ks_awgn_poi_linear = ks_awgn_poi_linear + np.max( np.abs( cum_sum_natural_hist - np.cumsum(cur_dataset_hist_awgn_poi_linear[:,channel]/normalize_size) ))
            ks_noiseflow = ks_noiseflow + np.max( np.abs( np.cumsum(cur_dataset_hist_natural_NF[:,channel]/normalize_size) - np.cumsum(cur_dataset_hist_noiseflow[:,channel]/normalize_size) ))
        ks_gan /= 3
        ks_gauss /= 3
        ks_awgn_poi_linear /= 3
        ks_noiseflow /=3
        print('KL gauss: {0:.4f}'.format(kl_gauss))
        print('KL awgn_poi_linear: {0:.4f}'.format(kl_awgn_poi_linear))
        print('KL noiseflow: {0:.4f}'.format(kl_noiseflow))
        print('KL gan: {0:.4f}'.format(kl_gan))
        print('---')
        print('KS gauss: {0:.4f}'.format(ks_gauss))
        print('KS awgn_poi_linear: {0:.4f}'.format(ks_awgn_poi_linear))
        print('KS noiseflow: {0:.4f}'.format(ks_noiseflow))
        print('KS gan: {0:.4f}'.format(ks_gan))
        print('========')
