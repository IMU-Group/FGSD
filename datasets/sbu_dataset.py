import cv2
from torch.utils.data import Dataset
import random
import os
import numpy as np
import torch
from collections import OrderedDict
from torchvision import transforms
from utils.transforms import JointRandHrzFlip, JointResize, \
                              JointNormalize, JointToTensor, \
                               JointRandVertFlip
                            

class SBUDataset(Dataset):
    def __init__(self,
                 data_root,
                 phase=None,
                 img_dirs=['ShadowImages'],
                 mask_dir='ShadowMasks',
                 augmentation=False,
                 im_size=400,
                 max_dataset_size=None,
                 normalize=True,
                 soft_mask_dir='ShadowMasks_soft'):
        
        self.root_dir = data_root
        self.img_dirs = img_dirs
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, img_dirs[0])))
        self.mask_dir = mask_dir
        self.augmentation = augmentation
        self.soft_mask_dir = soft_mask_dir
        
        self.size = len(self.img_names)
        # None means doesn't change the size of dataset to be loaded
        if max_dataset_size is not None:
            assert isinstance(max_dataset_size, int) and max_dataset_size > 0
            self.size = min(max_dataset_size, self.size)
            self.img_names = self.img_names[:self.size]
        
        assert phase in ['train', 'val', 'test', None]
        if phase == 'train':
            self.joint_transform = transforms.Compose([JointRandHrzFlip(),
                                                    #    JointRandVertFlip(),
                                                       JointResize(im_size)])
            img_transform = [ JointToTensor() ]
            if normalize:
                img_transform.append( JointNormalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225]) )  
            self.img_transform = transforms.Compose(img_transform)
            self.target_transform = transforms.ToTensor()

        elif phase in ['val', 'test']:
            self.joint_transform = None

            img_transform = [ JointToTensor() ]
            if normalize:
                img_transform.append( JointNormalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225]) )  
            self.img_transform = transforms.Compose(img_transform)

            self.target_transform = transforms.ToTensor()
            
        else: # pahse is None
            self.joint_transform = None
            self.img_transform = None
            self.target_transform = None

    def __getitem__(self, index):
        sample = OrderedDict()
        img_name = self.img_names[index]

        if self.augmentation:
            ret_key = ['ShadowImages_input']
            img_dir = random.choice(self.img_dirs)
            img_path = os.path.join(self.root_dir, img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ret_val= [ img ]


        else:
            ret_key = []
            ret_val = []
            for img_dir in self.img_dirs:
                img_path = os.path.join(self.root_dir, img_dir, img_name)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ret_key.append('ShadowImages_input')
                ret_val.append(img)
        self.endwith = '.jpg'
        mask_name = os.path.splitext(img_name)[0]+self.endwith
        mask_path = os.path.join(self.root_dir, self.mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask = ((cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 125)*255).astype(np.uint8)
        else:
            soft_mask_name = os.path.splitext(img_name)[0]+self.endwith
            soft_mask_path = os.path.join(self.root_dir, self.soft_mask_dir, soft_mask_name)
            mask_name = os.path.splitext(img_name)[0]+self.endwith
            mask_path = os.path.join(self.root_dir, self.soft_mask_dir, soft_mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Soft mask not found at path: {soft_mask_path}")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = ((mask > 25)*255).astype(np.uint8)
        ret_key.append('gt')
        ret_val.append(mask)

        # soft_mask
        soft_mask_name = os.path.splitext(img_name)[0]+self.endwith
        soft_mask_path = os.path.join(self.root_dir, self.soft_mask_dir, soft_mask_name)
        # print(soft_mask_path)
        soft_mask = cv2.imread(soft_mask_path, cv2.IMREAD_GRAYSCALE)
        ret_key.append('soft_gt')
        ret_val.append(soft_mask)
    
        if self.joint_transform:
            ret_val = self.joint_transform(ret_val)

        if self.img_transform:
            ret_val[0] = self.img_transform(ret_val[0])

        if self.target_transform:
            ret_val[-2] = self.target_transform(ret_val[-2])
            ret_val[-1] = self.target_transform(ret_val[-1])
        ret_key.append('im_name')
        ret_val.append(img_name)
        
        # print(ret_key)
        # print(ret_val)
        return OrderedDict(zip(ret_key, ret_val))


    def __len__(self):
        return self.size
    
    
    