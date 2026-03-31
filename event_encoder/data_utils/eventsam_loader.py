import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import os
import numpy as np
import cv2
from PIL import Image

class RGBEData(Dataset):
    def __init__(
        self,
        root,
        data=None,
        img_width=512,
        semantic_width=256,
        is_eval=False,
    ):
        self.root = root
        self.is_eval = is_eval
        
        self.data = data
        self.img_width = [img_width, img_width]
        self.semantic_width = [semantic_width, semantic_width]

        self.data_paths = [line.rstrip() for line in open(os.path.join(self.root, self.data))]

        print('The size of data is %d' % (len(self.data_paths)))
        self.image_pixel_mean =  torch.Tensor([0.485,0.456,0.406]).view(-1, 1, 1)
        self.image_pixel_std = torch.Tensor([0.229,0.224,0.225]).view(-1, 1, 1)
        self.evimg_pixel_mean = torch.Tensor([0.485,0.456,0.406]).view(-1, 1, 1)
        self.evimg_pixel_std = torch.Tensor([0.229,0.224,0.225]).view(-1, 1, 1)

    def __len__(self):
        return len(self.data_paths)

    def read_file_paths(self, index, is_eval=False):
        all_paths = self.data_paths[index]
        
        if len(all_paths.split(" ")) == 3:
            image_path, evimg_path, label_path = all_paths.split(" ")
        elif len(all_paths.split(" ")) == 2:
            image_path, evimg_path = all_paths.split(" ")
            label_path = None
        
        return image_path, evimg_path, label_path

    def load_label(self,path):
        label = Image.open(path)
        label = np.array(label)
        return label
    
    def __getitem__(self, index):
        image_path, evimg_path, label_path = self.read_file_paths(index, is_eval=self.is_eval)

        image = cv2.imread(image_path)
        evimg = cv2.imread(evimg_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        evimg = cv2.cvtColor(evimg, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image)
        evimg = F.to_tensor(evimg)
            
        image = torch.nn.functional.interpolate(
            image[None, :, :, :],
            size=self.img_width,
            mode='bilinear'
        )[0]
        evimg = torch.nn.functional.interpolate(
            evimg[None, :, :, :],
            size=self.img_width,
            mode='bilinear'
        )[0]
            
        image = (image - self.image_pixel_mean) / self.image_pixel_std
        evimg = (evimg - self.evimg_pixel_mean) / self.evimg_pixel_std

        return image, evimg
        

