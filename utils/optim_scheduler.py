from torch.optim import SGD, Adam, lr_scheduler

def get_parameters(model, base_lr, nbb_mult=1):
    if base_lr <= 0 or nbb_mult <= 0:
        raise ValueError("must be positive number")
    try:
        float(base_lr * 10)
        float(base_lr * nbb_mult)
    except OverflowError:
        raise ValueError("overflow the float range")
    bb_lr = []
    nbb_lr = []
    fcn_lr = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'backbone' in key:
            bb_lr.append(value)
        elif 'aux_layer' in key or 'upsample_proj' in key:
            fcn_lr.append(value)
        else:
            nbb_lr.append(value)

    params = [{'params': bb_lr, 'lr': base_lr},
              {'params': fcn_lr, 'lr': base_lr * 10},
              {'params': nbb_lr, 'lr': base_lr * nbb_mult}]
    params = [p for p in params if len(p['params']) != 0]
    return params


def get_optimizer(parameters, opt):
    if opt.optimizer == "SGD":
        return SGD(
            params=parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=False
        )
    elif opt.optimizer == "Adam":
        return Adam(
            params=parameters,
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
        )
    else:
        raise ValueError(f"Specified optimizer name '{opt.optimizer}' is not valid.")


def get_optim_scheduler(model, opt):
    params_group = get_parameters(model, opt.learning_rate)
    optimizer = get_optimizer(params_group, opt)

    # lambda_poly = lambda iter: pow((1.0 - iter / opt.max_iters), opt.power)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)     # too small
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs)     # Mean IOU: 0.180, Pixel ACC: 0.539

    return optimizer, scheduler
