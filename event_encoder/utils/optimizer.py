
import torch

def get_optimizer(params, ctx):
    if ctx.name == "Adam":
        return torch.optim.Adam(
            params=params,
            lr=ctx.lr
        )
    else:
        raise Exception(ctx.name + " optimizer is not supported.")