''' Frozen DeiT backbone that yields CLS-token features per transformer block. '''

import timm
import torch
import torch.nn as nn


class IntermediateDEIT(nn.Module):
    '''
        Wraps a pretrained timm DeiT/ViT so callers can probe each block's CLS token.

        Usage:
            deit = IntermediateDEIT("deit_small_patch16_224").cuda()
            for i, h_cls in deit.forward_intermediates(x):
                if i in probes:
                    probes[i].step_loss(h_cls.detach(), y)
    '''

    SUPPORTED = {
        "deit_tiny_patch16_224",
        "deit_small_patch16_224",
        "deit_base_patch16_224",
        "vit_tiny_patch16_224",
        "vit_small_patch16_224",
        "vit_base_patch16_224",
    }

    def __init__(self, model_name="deit_small_patch16_224", pretrained=True, checkpoint_path=None):
        super().__init__()
        if model_name not in self.SUPPORTED:
            raise ValueError(
                f"unsupported model {model_name!r}; pick from {sorted(self.SUPPORTED)}"
            )
        use_pretrained = pretrained and checkpoint_path is None
        self.vit = timm.create_model(model_name, pretrained=use_pretrained, num_classes=0)
        if checkpoint_path is not None:
            sd = torch.load(checkpoint_path, map_location="cpu")
            missing, unexpected = self.vit.load_state_dict(sd, strict=False)
            bad_missing = [k for k in missing if not k.startswith("head.")]
            if bad_missing or unexpected:
                raise RuntimeError(
                    f"checkpoint mismatch: missing={bad_missing} unexpected={unexpected}"
                )
            print(f"[IntermediateDEIT] loaded fine-tuned backbone from {checkpoint_path}")
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.vit.embed_dim
        self.depth = len(self.vit.blocks)
        self.model_name = model_name

    def train(self, mode=True):
        super().train(mode)
        self.vit.eval()
        return self

    @torch.no_grad()
    def forward_intermediates(self, x):
        '''
            Generator yielding (block_idx, cls_features) for each transformer block.
            cls_features: (B, embed_dim) — the CLS token after that block.
        '''
        vit = self.vit
        x = vit.patch_embed(x)
        x = vit._pos_embed(x)
        if hasattr(vit, "patch_drop"):
            x = vit.patch_drop(x)
        if hasattr(vit, "norm_pre"):
            x = vit.norm_pre(x)
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            yield i, x[:, 0, :]
