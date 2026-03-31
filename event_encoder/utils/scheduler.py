
import torch

def get_scheduler(optimizer, ctx):
    if ctx.name == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=ctx.gamma)
    else:
        raise Exception(ctx.name + " scheduler is not supported.")