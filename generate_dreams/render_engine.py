import torch

import generate_dreams.parametrizations as imparams


"""
Generates dream with model from a batch, with learning rate opt_lr, 
"""
def generate_dream(
    model:torch.nn.Module,
    batch,
    device,
    opt_lr=1e-3,
    iterations=(8,),
    parametrization="tanh",
    penal_f = None,
    penal_factor=0.0,
    limit_eps = 0.1
):
    # unpack batch to tensor images and labels, create y boolean onehot encoded tensor.
    x:torch.Tensor
    y:torch.Tensor
    x, y = batch

    x, y = x.clone().detach().to(device), y.clone().detach().to(device)

    out_shape = (len(iterations), x.shape[0], x.shape[1], x.shape[2], x.shape[3])

    images = torch.empty(size=out_shape, device=device)

    onehot_y = torch.nn.functional.one_hot(y, 10).to(torch.bool)

    prev_model_train_state = model.training
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    if parametrization == "linear":
        param_f = imparams.linear.linear_param(x, device=device)
    elif parametrization == "bound_hyperbolic":
        param_f = imparams.bound_hyperbolic.bound_hyperbolic_param(x, device=device, eps=limit_eps)
    else:
        param_f = imparams.hyperbolic.hyperbolic_param(x, device)


    params, image_f = param_f()
    optimizer = torch.optim.Adam(params, lr=opt_lr)


    with torch.no_grad():
        _initial_logits_all = model(image_f())
        _initial_target_logits = torch.masked_select(_initial_logits_all, onehot_y)
        _initial_other_logits = torch.masked_select(_initial_logits_all, ~onehot_y).reshape((x.shape[0], 9))

    num_im = 0
    for i in range(1,(iterations[-1] + 1)):
        def closure():
            optimizer.zero_grad()

            cur_img = image_f()

            out = model(cur_img)
            dream_logits = torch.masked_select(out, onehot_y)
            other_logits = torch.masked_select(out, ~(onehot_y)).reshape((x.shape[0], 9))

            _target_loss = torch.sub(_initial_target_logits, dream_logits)
            if torch.any(torch.le(_target_loss, -2)):
                # print('stopped opt')
                return 0

            if penal_f is not None:
                _other_targets_loss = penal_f(other_logits, _initial_other_logits)
                loss = (_target_loss + (_other_targets_loss * penal_factor)).mean()
            else:
                loss = _target_loss.mean()
            loss.backward()
            # print(loss)
            return loss

        optimizer.step(closure)
        if i in iterations:
            image = image_f()
            images[num_im] = image.clone().detach()
            num_im += 1

    model.train(mode=prev_model_train_state)

    if prev_model_train_state:
        for p in model.parameters():
            p.requires_grad_(mode=True)

    return images
