import torch
import torch.nn.functional as F

def loss_func(pred, gt, ctx, weight=None):
    if ctx.normalize:
        pred = torch.nn.functional.normalize(pred, p=2.0, dim=-1)
        gt = torch.nn.functional.normalize(gt, p=2.0, dim=-1)
        
    if ctx.name == "CosineSimilarity":
        loss = ctx.weight * (1 - torch.nn.CosineSimilarity(dim=ctx.dim)(pred, gt))
        
        if weight is not None:
            loss *= weight
    
        if ctx.reduction == "mean":
            return loss.mean()
        else:
            return loss
    elif ctx.name == "MSE":
        return ctx.weight * torch.nn.MSELoss(reduction=ctx.reduction)(pred, gt)
    elif ctx.name == "CrossEntropy":
        return ctx.weight * F.cross_entropy(pred, gt, reduction=ctx.reduction)
    else:
        raise Exception(ctx.name + " loss is not supported.")