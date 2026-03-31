import os, logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import DictConfig
import hydra

from data_utils.seal_loader import SEALData
from data_utils.collate import collate_fn_eval
from utils.log import print_results

from data_utils.data_class import DSEC11_CLASS, DSEC11_INSTANCE_ID, DSEC11_PRED_TO_ID
from data_utils.data_class import DSEC19_CLASS, DSEC19_INSTANCE_ID, DSEC19_PRED_TO_ID
from data_utils.data_class import DDD17_CLASS, DDD17_INSTANCE_ID, DDD17_PRED_TO_ID

from models.seal import SEAL

import numpy as np
from sklearn.metrics import average_precision_score


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / union if union > 0 else 0.0

def evaluate_class_for_iou(predictions, ground_truths, cls, iou_threshold):
    y_true = []
    y_score = []
    for pred, gt in zip(predictions, ground_truths):
        true_val = 1 if gt['label'] == cls else 0
        y_true.append(true_val)
        
        if pred['label'] == cls:
            if gt['label'] == cls:
                iou = compute_iou(pred['mask'], gt['mask'])
                score = 1.0 if iou >= iou_threshold else 0.0 # pred['confidence'] if iou >= iou_threshold else 0.0
            else:
                score = 1.0 # pred['confidence']
        else:
            score = 0.0
        y_score.append(score)
    
    if len(y_true) == 0:
        return np.nan
    return average_precision_score(np.array(y_true), np.array(y_score))

def compute_metrics_per_class(predictions, ground_truths, classes, class_name):
    results = {}
    for cls in classes:
        cls_name = class_name[cls]
        
        ap50 = evaluate_class_for_iou(predictions, ground_truths, cls, iou_threshold=0.5)
        ap25 = evaluate_class_for_iou(predictions, ground_truths, cls, iou_threshold=0.25)
        
        thresholds = np.arange(0.5, 1.0, 0.05)
        ap_list = []
        for thr in thresholds:
            ap = evaluate_class_for_iou(predictions, ground_truths, cls, iou_threshold=thr)
            ap_list.append(ap)
        overall_ap = np.mean(ap_list)
        
        results[cls_name] = {"AP50": ap50, "AP25": ap25, "AP": overall_ap}
        
    overall_AP = [vals["AP"] for vals in results.values() if not np.isnan(vals["AP"])]
    overall_AP = np.nanmean(overall_AP) if overall_AP else np.nan
    overall_AP50 = [vals["AP50"] for vals in results.values() if not np.isnan(vals["AP50"])]
    overall_AP50 = np.nanmean(overall_AP50) if overall_AP50 else np.nan
    overall_AP25 = [vals["AP25"] for vals in results.values() if not np.isnan(vals["AP25"])]
    overall_AP25 = np.nanmean(overall_AP25) if overall_AP25 else np.nan
        
    results["all_AP"] = overall_AP
    results["all_AP50"] = overall_AP50
    results["all_AP25"] = overall_AP25
    
    return results

@hydra.main(config_path="../configs", config_name="eval/DSEC11/seal.yaml")
def test(ctx: DictConfig):
    if ctx.dataset.test.name == 'DSEC11':
        DATA_CLASS = DSEC11_CLASS
        INSTANCE_ID = DSEC11_INSTANCE_ID
        PRED_TO_ID = DSEC11_PRED_TO_ID
    elif ctx.dataset.test.name == 'DSEC19':
        DATA_CLASS = DSEC19_CLASS
        INSTANCE_ID = DSEC19_INSTANCE_ID
        PRED_TO_ID = DSEC19_PRED_TO_ID
    elif ctx.dataset.test.name == 'DDD17':
        DATA_CLASS = DDD17_CLASS
        INSTANCE_ID = DDD17_INSTANCE_ID
        PRED_TO_ID = DDD17_PRED_TO_ID

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    SEALDataset = SEALData(
        name=ctx.dataset.test.name,
        root=ctx.dataset.test.dir,
        data=ctx.dataset.test.data,
        gt=ctx.dataset.test.gt,
        img_width=ctx.dataset.test.img_width,
        semantic_width=ctx.dataset.test.semantic_width,
        is_eval=True
    )
    SEALLoader = torch.utils.data.DataLoader(
        dataset=SEALDataset, 
        batch_size=ctx.dataset.test.batch_size,
        collate_fn=collate_fn_eval,
        shuffle=ctx.dataset.test.shuffle
    )

    """ Build model """
    CHECKPOINT_PATH = ctx.general.checkpoint
    
    print("Evaluating ", CHECKPOINT_PATH, "...")

    seal = SEAL(ctx.model).to(device)

    total_params = sum(p.numel() for p in seal.parameters())
    print("Total parameters:", total_params)
            
    seal.eval()
    seal.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    
    text_classifier = torch.load(ctx.dataset.test.classifier).to(device).float()
    
    predictions = []
    ground_truths = []

    for images, evimgs, semantic_label, instance_mask, instance_label in tqdm(SEALLoader):            
        evimgs = evimgs.to(device)[0]
        instance_mask = instance_mask[0].to(device)
        instance_label = instance_label[0].to(device)
            
        seal.set_image(evimgs)
        for mask, label in zip(instance_mask, instance_label):
            with torch.no_grad():
                if ctx.dataset.test.prompt == "point":
                    points, point_labels = seal.generate_prompt(
                        mask.cpu().numpy(),
                        type="point",
                        point_num=3, 
                        sample="furthest"
                    )
                    pred_masks, pred_labels, pred_scores, pred_idx = seal.predict(
                        input_points=points, 
                        input_labels=point_labels,
                        input_box=None,
                        fusion=text_classifier[INSTANCE_ID, :]
                    )
                elif ctx.dataset.test.prompt == "box":
                    box = seal.generate_prompt(
                        mask.cpu().numpy(),
                        type="box"
                    )
                    pred_masks, pred_labels, pred_scores, pred_idx = seal.predict(
                        input_points=None, 
                        input_labels=None,
                        input_box=box,
                        fusion=text_classifier[INSTANCE_ID, :]
                    )

            if pred_idx == -1:
                continue
                
            pred_mask = pred_masks[pred_idx]
            pred_label = pred_labels[pred_idx]
            pred_score = pred_scores[pred_idx]
            
            predictions.append(
                {
                    'mask': pred_mask.cpu().numpy(), 
                    'label': PRED_TO_ID[pred_label.item()], 
                    'confidence': pred_score.item()
                }
            )
            ground_truths.append({'mask': mask.cpu().numpy(), 'label': label.item()})
                
    class_ap_results = compute_metrics_per_class(predictions, ground_truths, INSTANCE_ID, DATA_CLASS)
    print_results(class_ap_results, [DATA_CLASS[i] for i in INSTANCE_ID])

if __name__ == '__main__':
    test()

