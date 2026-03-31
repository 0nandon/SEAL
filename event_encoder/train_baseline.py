
import os, logging
import numpy as np
from sklearn.metrics import average_precision_score

import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from loss.mse_loss import MSELoss

from models.baseline import BASELINE
from data_utils.baseline_loader import BaselineData
from data_utils.collate import collate_fn_train_baseline, collate_fn_eval_baseline
from data_utils.data_class import DSEC11_CLASS, DSEC11_INSTANCE_ID, DSEC11_PRED_TO_ID
from data_utils.data_class import DSEC19_CLASS, DSEC19_INSTANCE_ID, DSEC19_PRED_TO_ID
from data_utils.data_class import DDD17_CLASS, DDD17_INSTANCE_ID, DDD17_PRED_TO_ID

from utils.log import log_func, print_results_log
from utils.loss import loss_func
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler

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
                score = 1.0 if iou >= iou_threshold else 0.0
            else:
                score = 1.0
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

def eval(baseline, iteration, best_iter, best_ap, ctx, log):
    log.info("eval start!")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if ctx.name == 'DSEC11':
        DATA_CLASS = DSEC11_CLASS
        INSTANCE_ID = DSEC11_INSTANCE_ID
        PRED_TO_ID = DSEC11_PRED_TO_ID
    elif ctx.name == 'DSEC19':
        DATA_CLASS = DSEC19_CLASS
        INSTANCE_ID = DSEC19_INSTANCE_ID
        PRED_TO_ID = DSEC19_PRED_TO_ID
    elif ctx.name == 'DDD17':
        DATA_CLASS = DDD17_CLASS
        INSTANCE_ID = DDD17_INSTANCE_ID
        PRED_TO_ID = DDD17_PRED_TO_ID
    
    baseline.eval()
    
    """ Build dataloaders """    
    BaselineDataset = BaselineData(
        name=ctx.name,
        root=ctx.dir,
        data=ctx.data,
        gt=ctx.gt,
        img_width=ctx.img_width,
        is_eval=True,
        type=ctx.type
    )
    BaselineLoader = torch.utils.data.DataLoader(
        dataset=BaselineDataset, 
        batch_size=ctx.batch_size,
        collate_fn=collate_fn_eval_baseline,
        shuffle=ctx.shuffle
    )
    
    text_classifier = torch.load(ctx.classifier).to(device).float()

    predictions = []
    ground_truths = []

    for event_representation, evimg, semantic_label, instance_mask, instance_label in tqdm(BaselineLoader):            
        event_representation = event_representation[0].to(device)
        evimg = evimg[0].to(device)
            
        instance_mask = instance_mask[0].to(device)
        instance_label = instance_label[0].to(device)
            
        baseline.set_image(evimg)
        for mask, label in zip(instance_mask, instance_label):
            with torch.no_grad():
                if ctx.prompt == "point":
                    points, point_labels = baseline.generate_prompt(
                        mask.cpu().numpy(),
                        type="point",
                        point_num=3, 
                        sample="furthest"
                    )
                    pred_masks, pred_labels, pred_scores, pred_idx = baseline.predict(
                        event_representation=event_representation,
                        input_points=points, 
                        input_labels=point_labels,
                        input_box=None,
                        classifier=text_classifier[INSTANCE_ID, :]
                    )
                elif ctx.prompt == "box":
                    box = baseline.generate_prompt(
                        mask.cpu().numpy(),
                        type="box"
                    )
                    pred_masks, pred_labels, pred_scores, pred_idx = baseline.predict(
                        event_representation=event_representation,
                        input_points=None, 
                        input_labels=None,
                        input_box=box,
                        classifier=text_classifier[INSTANCE_ID, :]
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
    print_results_log(class_ap_results, [DATA_CLASS[i] for i in INSTANCE_ID], log)
    
    baseline.train()

    return (iteration, class_ap_results["all_AP"]) if class_ap_results["all_AP"] >= best_ap else (best_iter, best_ap)


@hydra.main(config_path="../configs_v2", config_name="seal_v2.yaml")
def run(ctx: DictConfig):
    log = log_func(os.path.join(ctx.log.dir, ctx.log.name))
    log.info('=====Training script for event_vit_encoder')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    best_ap = 0.
    best_iter = 0.
    
    """ Build dataloaders """
    BaselineDataset = BaselineData(
        name=ctx.dataset.train.name,
        root=ctx.dataset.train.dir,
        data=ctx.dataset.train.data,
        img_width=ctx.dataset.train.img_width,
        augmentation=True,
        type=ctx.dataset.train.type
    )
    BaselineLoader = torch.utils.data.DataLoader(
        dataset=BaselineDataset, 
        batch_size=ctx.dataset.train.batch_size,
        collate_fn=collate_fn_train_baseline,
        shuffle=ctx.dataset.train.shuffle
    )

    """ Build model """
    baseline = BASELINE(ctx.model).to(device)
    
    total_params = sum(p.numel() for p in baseline.parameters())
    trainable_params = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print("Total parameters:", total_params)
    print("Total trainable parameters: ", trainable_params)
    
    train_para_name_list = []
    train_para_list = []
    for name, param in baseline.named_parameters():
        if param.requires_grad:
            train_para_name_list.append(name)
            train_para_list.append(param)

    log.info("=====Train Params: " + str(train_para_name_list))
    
    optimizer = get_optimizer(
        params=[{
            'params': train_para_list
            }], 
        ctx=ctx.general.optimizer
    )
    scheduler = get_scheduler(optimizer=optimizer, ctx=ctx.general.scheduler)

    iteration = 0
    baseline.train()
    
    loss_sum = 0.0
    for epoch in range(ctx.general.epoch):
        log.info('=====epoch ' + str(epoch))
        for event_representation, semantic_label, instance_mask_gt, instance_img_gt in BaselineLoader:
            event_representation = event_representation.to(device)
            
            instance_mask_gt = [mask.to(device) for mask in instance_mask_gt]
            instance_img_gt = [img.to(device) for img in instance_img_gt]
            
            optimizer.zero_grad()

            loss = baseline(
                event_representation,
                instance_mask_gt,
                instance_img_gt,
            )
            
            loss['total'].backward()
            optimizer.step()
            loss_sum += loss['total'].item()
            
            iteration += 1 
            if iteration % ctx.general.iter_log == 0:
                log.info('iteration: ' + str(iteration) + ' loss: ' + str(loss_sum / ctx.general.iter_log))
                loss_sum = 0.0
            
            if iteration >= ctx.general.eval_start and iteration % ctx.general.eval_check == 0:            
                best_iter, best_ap = eval(
                    baseline, 
                    iteration,
                    best_iter,
                    best_ap,
                    ctx.dataset.test1,
                    log
                )
                log.info("BEST ITER: {}\n".format(best_iter))
                log.info("")
                
                if best_iter == iteration:
                    torch.save(baseline.state_dict(), os.path.join(ctx.general.checkpoint, "best_checkpoint_{}.pt".format(iteration)))

                _, _ = eval(
                    baseline, 
                    iteration,
                    best_iter,
                    best_ap,
                    ctx.dataset.test2,
                    log
                )
                log.info("")

        if epoch % ctx.general.scheduler.step == 0:
            scheduler.step()

if __name__ == '__main__':
    run()

