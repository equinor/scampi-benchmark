import torch
import vision_transformer as vits
import vision_transformer_mae as vits_mae
import utils

def load_dino_model(args):
    # Load DINOv1 model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    return model

def load_dinov2_model(args):
    # Load DINOv2 model
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    if args.arch == 'vit_small':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif args.arch == 'vit_base':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    else:
        raise ValueError(f"Architecture {args.arch} not supported with DINOv2 weights)")
    return model

def load_vit_mae_model(args):
    # Load ViT-MAE model
    if args.arch != 'vit_base':
        raise ValueError("Unsupported architecture. Only 'vit_base' is supported.")
    
    model = vits_mae.__dict__['vit_base_patch16'](num_classes=0, drop_path_rate=0.1, global_pool=True)
    url = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    interpolate_pos_embed(model, checkpoint_model)
    
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Missing keys when loading pretrained weights: {msg.missing_keys}")
    return model

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
