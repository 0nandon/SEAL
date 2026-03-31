
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import os
import numpy as np
import cv2
from PIL import Image
import re
import clip
import random
from data_utils.data_class import DSEC11_INSTANCE_ID, DSEC19_INSTANCE_ID, DDD17_INSTANCE_ID

class SEALData(Dataset):
    def __init__(
        self,
        name,
        root,
        data=None,
        gt=None,
        img_width=512,
        semantic_width=256,
        is_eval=False,
        augmentation=False,
        mask_levels=['l', 'm', 's']
    ):
        self.name = name
        self.root = root
        self.is_eval = is_eval
        self.augmentation = augmentation
        self.mask_levels = mask_levels
        
        self.data = data
        self.gt = gt
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

    def read_file_paths(self, index):
        all_paths = self.data_paths[index]
        image_path, evimg_path, label_path = all_paths.split(" ")
        return image_path, evimg_path, label_path
    
    def get_text_feat(self, text):
        text = re.sub(r'\d+', '', text)
        run_on_gpu = torch.cuda.is_available()

        with torch.no_grad():
            text = clip.tokenize(text, truncate=True)  #tokenize
            if run_on_gpu:
                text = text.cuda()
            
        text_embedding = self.clip_model.encode_text(text)  #embed with text encoder
        return text_embedding

    def load_img(self, image_path, evimg_path):
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
        
    def process_mask_cache(self, mask_cache, augmentation=False):
        instance_mask_gt = {}
        instance_mask_x2_gt = {}
        instance_caption_short_gt = {}
        instance_caption_long_gt = {}
        instance_img_gt = {}
        instance_weight = {}
        
        for lvl in self.mask_levels: # ['l', 'm', 's']:
            mask_img = mask_cache[lvl]['mask']
            mask_idx = np.unique(mask_img)

            caption_feat_short = []
            caption_feat_long = []
            img_feat = []
            weight = []
            masks = []
            i = 0
            for idx in mask_idx:
                if idx == -1:
                    continue
                
                mask = mask_img == idx
                if mask.sum() == 0:
                    continue
                
                caption_feat_short.append(mask_cache[lvl]['feat']['short'][i])
                caption_feat_long.append(mask_cache[lvl]['feat']['long'][i])
                img_feat.append(mask_cache[lvl]['feat']['img'][i].cpu())
                
                weight.append(1.)
                masks.append(torch.from_numpy(mask))
                i += 1

            caption_feat_short = torch.stack(caption_feat_short, dim=0)
            caption_feat_long = torch.stack(caption_feat_long, dim=0)
            img_feat = torch.stack(img_feat, dim=0)
            
            weight = torch.tensor(weight)
            
            masks = torch.stack(masks, dim=0)

            masks_x2 = torch.nn.functional.interpolate(
                masks[None, :, :, :].float(),
                size=self.img_width[0],
                mode='nearest'
            )[0].bool()
            
            masks = torch.nn.functional.interpolate(
                masks[None, :, :, :].float(),
                size=self.semantic_width,
                mode='nearest'
            )[0].bool()

            new_caption_feat_short = []
            new_caption_feat_long = []
            new_img_feat = []
            
            new_weight = []
            new_masks = []
            new_masks_x2 = []
            for i, mask in enumerate(masks):
                if mask.sum() == 0:
                    continue
                
                new_caption_feat_short.append(caption_feat_short[i])
                new_caption_feat_long.append(caption_feat_long[i])
                new_img_feat.append(img_feat[i])
                
                new_weight.append(weight[i])
                
                if augmentation:
                    new_masks.append(torch.flip(mask, [1]))
                    new_masks_x2.append(torch.flip(masks_x2[i], [1]))
                else:   
                    new_masks.append(mask)
                    new_masks_x2.append(masks_x2[i])
                
            new_caption_feat_short = torch.stack(new_caption_feat_short)
            new_caption_feat_long = torch.stack(new_caption_feat_long)
            new_img_feat = torch.stack(new_img_feat)
            
            new_weight = torch.tensor(new_weight)
            
            new_masks = torch.stack(new_masks)
            new_masks_x2 = torch.stack(new_masks_x2)

            instance_mask_gt[lvl] = new_masks
            instance_mask_x2_gt[lvl] = new_masks_x2
            instance_caption_short_gt[lvl] = new_caption_feat_short.float()
            instance_caption_long_gt[lvl] = new_caption_feat_long.float()
            instance_img_gt[lvl] = new_img_feat.float()
            instance_weight[lvl] = new_weight.float()
        
        return instance_mask_gt, instance_mask_x2_gt, instance_caption_short_gt, instance_caption_long_gt, instance_img_gt, instance_weight

    def load_label(self,path):
        label = Image.open(path)
        label = np.array(label)
        return label
    
    def get_instance_gt(self, masks, semantic_label):
        mask_gt = []
        label_gt = []
        for mask in masks:
            mask = mask != -1
            labels = semantic_label[mask]
            label = np.bincount(labels).argmax()

            if self.name == "DSEC11":
                if label not in DSEC11_INSTANCE_ID:
                    continue
            elif self.name == "DSEC19":
                if label not in DSEC19_INSTANCE_ID:
                    continue
            elif self.name == "DDD17":
                if label not in DDD17_INSTANCE_ID:
                    continue
            
            mask_gt.append(mask)
            label_gt.append(int(label))
            
        mask_gt = torch.stack(mask_gt)
        label_gt = torch.tensor(label_gt)
        mask_gt = torch.nn.functional.interpolate(
            mask_gt[None, :, :, :].float(),
            size=self.img_width[0], 
            mode='nearest'
        )[0].bool()
            
        return mask_gt, label_gt
    
    def __getitem__(self, index):
        image_path, evimg_path, label_path = self.read_file_paths(index)
        image, evimg = self.load_img(image_path, evimg_path)
        semantic_label = torch.from_numpy(self.load_label(label_path))

        if self.is_eval:
            file_name = os.path.basename(image_path)[:-4]
            parent_dir = os.path.dirname(image_path)
            mask_path = os.path.join(parent_dir, "../" + self.gt, file_name + ".pt")
            masks = torch.load(mask_path)
            instance_mask, instance_label = self.get_instance_gt(masks, semantic_label)

            return image, evimg, semantic_label, instance_mask, instance_label
        else:
            file_name = os.path.basename(image_path)[:-4]
            parent_dir = os.path.dirname(image_path)
            mask_path = os.path.join(parent_dir, "../mhsg", file_name + ".pt")
            mask_cache = torch.load(mask_path)
            
            if self.augmentation:
                value_flip = round(random.random())
                if value_flip > 0.5:
                    evimg = torch.flip(evimg, [2])

            (
                instance_mask_gt, 
                instance_mask_x2_gt, 
                instance_caption_short_gt, 
                instance_caption_long_gt,
                instance_img_gt,
                instance_weight
            )= self.process_mask_cache(mask_cache, augmentation=self.augmentation and (value_flip > 0.5))

            return (
                image,
                evimg, 
                semantic_label, 
                instance_mask_gt, 
                instance_mask_x2_gt, 
                instance_caption_short_gt, 
                instance_caption_long_gt,
                instance_img_gt,
                instance_weight
            )
