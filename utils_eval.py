import os
import torch
import vision_transformer as vits
import vision_transformer_mae as vits_mae

def load_model(args):
    if os.path.isfile(args.pretrained_weights):
        model = load_from_local_file(args)
    elif args.pretrained_weights == 'scampi_dino':
        model = load_scampi_model(args.arch, args.patch_size)
    elif args.pretrained_weights == 'dinov1':
        model = load_dinov1_model(args.arch, args.patch_size)
    elif args.pretrained_weights == 'dinov2':
        model = load_dinov2_model(args)
    elif args.pretrained_weights == 'vit_mae':
        model = load_vit_mae_model(args)
    else:
        model = vits.__dict__[args.model_name](patch_size=args.patch_size, num_classes=0)
        print("Using random weights.")
    return model

def load_from_local_file(args):
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    state_dict = torch.load(args.pretrained_weights, map_location="cpu")
    if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
        print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[args.checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    return model

def load_from_url(model, url):
    state_dict = torch.hub.load_state_dict_from_url(url=url)
    model.load_state_dict(state_dict, strict=True)

def load_scampi_model(arch, patch_size):
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    if arch == 'vit_small' and patch_size == 16:
        #url = 'https://huggingface.co/IverMartinsen/scampi-dino-vits16/resolve/main/vit_small_backbone.pth?download=true'
        url = 'https://huggingface.co/IverMartinsen/scampi-test/resolve/main/vit_small_backbone_random_weights.pth?download=true'
    elif arch == 'vit_base' and patch_size == 16:
        url = 'https://huggingface.co/IverMartinsen/scampi-dino-vitb16/resolve/main/vit_base_backbone.pth?download=true'
    else:
        raise ValueError(f"Unsupported architecture {arch} with patch size {patch_size}.")
    print("Load SCAMPI pretrained weights.")
    load_from_url(model, url)
    return model

def load_dinov1_model(model_name, patch_size):
    model = vits.__dict__[model_name](patch_size=patch_size, num_classes=0)
    url = None
    if model_name == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif model_name == "vit_small" and patch_size == 8:
        url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
    elif model_name == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif model_name == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    elif model_name == "xcit_small_12_p16":
        url = "dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain.pth"
    elif model_name == "xcit_small_12_p8":
        url = "dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain.pth"
    elif model_name == "xcit_medium_24_p16":
        url = "dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain.pth"
    elif model_name == "xcit_medium_24_p8":
        url = "dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain.pth"
    elif model_name == "resnet50":
        url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"
    if url is not None:
        print("Load DINOv1 pretrained weights.")
        load_from_url(model, "https://dl.fbaipublicfiles.com/dino/" + url)
    else:
        print("There is no reference weights available for this model => We use random weights.")
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
        raise ValueError("Unsupported architecture. Only 'vit_base' is supported for ViT-MAE.")
    
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
