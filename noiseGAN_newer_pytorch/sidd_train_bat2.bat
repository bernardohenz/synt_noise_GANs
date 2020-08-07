@echo [off]

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_00400/IP/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_00400/IP/both_datasets/00400 --name SIDD_cleanTo400_IP_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 1.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_00400/N6/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_00400/N6/both_datasets/00400 --name SIDD_cleanTo400_N6_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 1.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_00400/S6/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_00400/S6/both_datasets/00400 --name SIDD_cleanTo400_S6_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 1.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_00800/G4/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_00800/G4/both_datasets/00800 --name SIDD_cleanTo800_G4_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 2.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_00800/GP/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_00800/GP/both_datasets/00800 --name SIDD_cleanTo800_GP_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 2.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_00800/IP/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_00800/IP/both_datasets/00800 --name SIDD_cleanTo800_IP_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 2.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_00800/N6/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_00800/N6/both_datasets/00800 --name SIDD_cleanTo800_N6_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 2.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_00800/S6/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_00800/S6/both_datasets/00800 --name SIDD_cleanTo800_S6_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 2.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_01600/G4/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_01600/G4/both_datasets/01600 --name SIDD_cleanTo1600_G4_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_01600/GP/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_01600/GP/both_datasets/01600 --name SIDD_cleanTo1600_GP_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_01600/IP/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_01600/IP/both_datasets/01600 --name SIDD_cleanTo1600_IP_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_01600/N6/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_01600/N6/both_datasets/01600 --name SIDD_cleanTo1600_N6_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

# python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_01600/S6/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_01600/S6/both_datasets/01600 --name SIDD_cleanTo1600_S6_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_03200/G4/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_03200/G4/both_datasets/03200 --name SIDD_cleanTo3200_G4_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_03200/GP/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_03200/GP/both_datasets/03200 --name SIDD_cleanTo3200_GP_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_03200/IP/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_03200/IP/both_datasets/03200 --name SIDD_cleanTo3200_IP_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_03200/N6/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_03200/N6/both_datasets/03200 --name SIDD_cleanTo3200_N6_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 3.5

python train.py --dataroot_A /media/bernardo/Storage/SIDD_patches_03200/S6/both_datasets/gts --dataroot_B /media/bernardo/Storage/SIDD_patches_03200/S6/both_datasets/03200 --name SIDD_cleanTo3200_S6_new_pytorch --model cycle_gan --no_dropout --lambda_std 10 --lambda_low_freq 10 --n_epochs 50 --n_epochs_decay 0 --compare_noisy_versions std_high --low_pass_std 4.5

