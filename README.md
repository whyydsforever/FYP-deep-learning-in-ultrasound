# FYP-deep-learning-in-ultrasound
This project is for the FYP of NUS Suzhou Research Institute. My project is Deep learning in Medical Ultrasound Images.<br>
The dataset is BUSI dataset, which is avalibale at https://scholar.cu.edu.eg/?q=afahmy/pages/dataset <br>
Our project used the Pix2PixHD model and made some adjustments. The offical Pix2PixHD model is at https://github.com/NVIDIA/pix2pixHD. And the adjustments are in the Pix2PixHD folder. <br>
The usage of FID Score is avaliable at https://github.com/mseitzer/pytorch-fid.<br>
## Training
To train Pix2PixHD, use the command:<br> `<python train.py --name [project name]  --label_nc 0 --no_instance --gpu_ids 0 --dataroot [path to dataset] --netG local --gpu_ids 0 --ngf 32  --niter 50 --niter_decay 50>`
