import torch
import torch.nn as nn

def load_weight(model:nn.Module, weight_dir:str, strict = True) -> nn.Module:
    current_state_dict = model.state_dict()
    state_dict = torch.load(weight_dir)
    for key in state_dict:
        if key not in list(current_state_dict.keys()):
            continue

        if current_state_dict[key].shape != state_dict[key].shape:
            if strict:
                raise ValueError(
                    "%s shape mismatch (src=%s, target=%s)"
                    % (
                        key,
                        list(state_dict[key].shape),
                        list(current_state_dict[key].shape),
                    )
                )
            else:
                print(
                    "Skip loading %s due to shape mismatch (src=%s, target=%s)"
                    % (
                        key,
                        list(state_dict[key].shape),
                        list(current_state_dict[key].shape),
                    )
                )
        else:
            current_state_dict[key].copy_(state_dict[key])
    model.load_state_dict(current_state_dict)

    unused_keys = list(state_dict.keys())
    unused_params = []
    for name, param in model.named_parameters():
        if name in state_dict:
            # param.requires_grad = False
            unused_keys.remove(name)
        else:
            unused_params.append(name)
    unused_keys = [each for each in unused_keys if (("running_mean" not in each) and ("running_var" not in each) and ("num_batches_tracked" not in each))]

    print('Unused keys in pretrained_weights:', unused_keys)
    print('Parameters in model without loaded weights:', unused_params)