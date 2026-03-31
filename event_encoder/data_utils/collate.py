
import torch

def collate_fn(batch):
    image, evimg, semantic_label, instance_mask, instance_mask_x2, instance_caption_short, instance_caption_long, instance_img, instance_weight = zip(*batch)
    
    image = torch.stack(image, dim=0)
    evimg = torch.stack(evimg, dim=0)

    return image, evimg, semantic_label, instance_mask, instance_mask_x2, instance_caption_short, instance_caption_long, instance_img, instance_weight


def collate_fn_eval(batch):
    image, evimg, semantic_label, instance_mask, instance_label = zip(*batch)
    
    image = torch.stack(image, dim=0)
    evimg = torch.stack(evimg, dim=0)
    semantic_label = torch.stack(semantic_label, dim=0)

    return image, evimg, semantic_label, instance_mask, instance_label


def collate_fn_eval_part(batch):
    image, evimg, instance_mask, instance_label, file_name = zip(*batch)
    
    image = torch.stack(image, dim=0)
    evimg = torch.stack(evimg, dim=0)

    return image, evimg, instance_mask, instance_label, file_name