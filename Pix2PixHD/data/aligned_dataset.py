import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import albumentations as A
import random
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:                              
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        ### input A (label maps)
        A_path = self.A_paths[index]              
        A = Image.open(A_path).convert('RGB')        
        params = get_params(self.opt, A.size)
       
       
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params, grayscale=(self.opt.input_nc == 1)) # for input_nc == 1
            A_tensor = transform_A(A)
        else:
            transform_A = get_transform(self.opt, params, grayscale=False, method=Image.NEAREST, normalize=False) # for input_nc == 1
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params, grayscale=(self.opt.output_nc == 1)) # for output_nc == 1      
            B_tensor = transform_B(B)
            
            if random.random()>0.5:  
             A_tensor,B_tensor=self.aug(A_tensor,B_tensor)
        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]            
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))
        
                                      
       
        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

    def aug(self,image,mask):
        image_np=np.array(image)
        mask_np=np.array(mask)
        transform=A.OneOf([
            
            A.ShiftScaleRotate(scale_limit=0,rotate_limit=0,p=1),
            A.ShiftScaleRotate(shift_limit=0,rotate_limit=0,p=1),
           
            
        ],p=1)
        transformed = transform(image=image_np, mask=mask_np)
        image_aug=transformed["image"]
        mask_aug=transformed["mask"]
        return image_aug,mask_aug    
