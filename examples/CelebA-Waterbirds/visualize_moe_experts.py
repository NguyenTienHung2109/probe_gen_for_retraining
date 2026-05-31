''' Per-expert Grad-CAM heatmaps for a MoE head on a frozen DeiT backbone.

    Auto-detects ckpt format:
      - "old"    probe ckpt (probe_block{i}.pth, has block_idx)            → needs --backbone-checkpoint (DeiT backbone-only)
      - "stage1" combined  ckpt (deit_small_moe_waterbirds.pth)            → backbone loaded from same file
      - "stage2" MoE-only  ckpt (deit_small_moe_finetuned.pth)             → backbone from --backbone-checkpoint (defaults to config["stage1_ckpt"])

    Picks N images per Waterbirds group from test split, runs Grad-CAM on patch
    tokens at the probed block (target = each expert's logit contribution to the
    predicted class), saves one PNG per sample with input + per-expert heatmaps.

    Usage:
        # Stage 1 ckpt
        PYTHONPATH=. python examples/CelebA-Waterbirds/visualize_moe_experts.py \
            --probe-ckpt checkpoints/deit_small_moe_waterbirds.pth \
            --data-root ./data/waterbirds/waterbird_complete95_forest2water2 \
            --out-dir outputs/expert_heatmaps/stage1 --n-per-group 2

        # Stage 2 ckpt (will auto-load backbone from config["stage1_ckpt"])
        PYTHONPATH=. python examples/CelebA-Waterbirds/visualize_moe_experts.py \
            --probe-ckpt checkpoints/deit_small_moe_finetuned.pth \
            --data-root ./data/waterbirds/waterbird_complete95_forest2water2 \
            --out-dir outputs/expert_heatmaps/stage2 --n-per-group 2
'''

import argparse
import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from intermediate_gen.datasets import MyWaterBirdsDataset
from intermediate_gen import MoEProbeModel


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
    ])


def denormalize(t):
    img = t.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def detect_format(ckpt):
    if "backbone_state" in ckpt and "moe_state" in ckpt:
        return "stage1"
    if "state_dict" in ckpt and "block_idx" in ckpt:
        return "old"
    if "state_dict" in ckpt:
        return "stage2"
    raise ValueError(f"unknown ckpt format, keys: {list(ckpt.keys())}")


def load_components(probe_ckpt_path, backbone_ckpt_path, device):
    ckpt = torch.load(probe_ckpt_path, map_location=device)
    fmt = detect_format(ckpt)
    cfg = ckpt["config"]

    if fmt == "stage1":
        backbone_name = cfg["backbone"]
        backbone_state = ckpt["backbone_state"]
        moe_state = ckpt["moe_state"]
        moe_kwargs = dict(
            input_size=cfg["embed_dim"],
            output_size=cfg["num_classes"],
            num_experts=cfg["num_experts"],
            optimizer="adam",
        )
        apply_final_norm = True
        block_idx = None
        if backbone_ckpt_path:
            print(f"[warn] --backbone-checkpoint ignored for Stage-1 ckpt")
    elif fmt == "stage2":
        backbone_name = cfg["backbone"]
        bb_path = backbone_ckpt_path or cfg.get("stage1_ckpt")
        if not bb_path or not Path(bb_path).exists():
            raise ValueError(
                f"Stage-2 ckpt needs --backbone-checkpoint pointing to Stage-1 ckpt; "
                f"config['stage1_ckpt']={cfg.get('stage1_ckpt')!r} not found on disk"
            )
        bb_ckpt = torch.load(bb_path, map_location=device)
        if "backbone_state" not in bb_ckpt:
            raise ValueError(f"--backbone-checkpoint {bb_path} is not a Stage-1 ckpt")
        backbone_state = bb_ckpt["backbone_state"]
        moe_state = ckpt["state_dict"]
        moe_kwargs = dict(
            input_size=cfg["input_size"],
            output_size=cfg["output_size"],
            num_experts=cfg["num_experts"],
            lambda_div=cfg.get("lambda_div", 0.0),
            lambda_sp=cfg.get("lambda_sp", 0.0),
            lambda_bal=cfg.get("lambda_bal", 0.0),
            bal_beta=cfg.get("bal_beta", 0.99),
            optimizer="adam",
        )
        apply_final_norm = True
        block_idx = None
    else:  # "old" probe format
        backbone_name = cfg.get("backbone", "deit_small_patch16_224")
        if not backbone_ckpt_path:
            raise ValueError("old probe ckpt requires --backbone-checkpoint (DeiT backbone-only)")
        backbone_state = torch.load(backbone_ckpt_path, map_location=device)
        moe_state = ckpt["state_dict"]
        moe_kwargs = dict(
            input_size=cfg["input_size"],
            output_size=cfg["output_size"],
            num_experts=cfg["num_experts"],
            lambda_div=cfg.get("lambda_div", 0.0),
            lambda_sp=cfg.get("lambda_sp", 0.0),
            lambda_bal=cfg.get("lambda_bal", 0.0),
            bal_beta=cfg.get("bal_beta", 0.99),
            optimizer="adam",
        )
        apply_final_norm = False
        block_idx = ckpt["block_idx"]

    vit = timm.create_model(backbone_name, pretrained=False, num_classes=0)
    missing, unexpected = vit.load_state_dict(backbone_state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("head.")]
    if bad_missing:
        print(f"[backbone] WARNING missing keys: {bad_missing}")
    if unexpected:
        ignorable = [k for k in unexpected if k.startswith("head.")]
        rest = [k for k in unexpected if not k.startswith("head.")]
        if rest:
            print(f"[backbone] WARNING unexpected keys: {rest}")
    vit.eval().to(device)
    for p in vit.parameters():
        p.requires_grad_(False)

    if block_idx is None:
        block_idx = len(vit.blocks) - 1

    moe = MoEProbeModel(**moe_kwargs)
    missing, unexpected = moe.load_state_dict(moe_state, strict=False)
    if unexpected:
        print(f"[moe] WARNING unexpected keys: {unexpected}")
    if missing and any(not k.startswith("balance_loss_fn.") for k in missing):
        print(f"[moe] WARNING missing keys: {missing}")
    moe.eval().to(device)
    for p in moe.parameters():
        p.requires_grad_(False)

    return vit, moe, block_idx, apply_final_norm, fmt, cfg


def select_samples(base_dataset, n_per_group, seed):
    rng = np.random.RandomState(seed)
    test_idxs = base_dataset.test_idxs
    test_groups = base_dataset.group_array[test_idxs]
    selected = []
    for g in range(base_dataset.n_groups):
        pool = test_idxs[test_groups == g]
        if len(pool) == 0:
            continue
        n = min(n_per_group, len(pool))
        chosen = rng.choice(pool, size=n, replace=False)
        for idx in chosen:
            selected.append((int(idx), int(g), int(base_dataset.y_array[idx])))
    return selected


def expert_gradcam(vit, probe, img, block_idx, apply_final_norm, device, grid_size=14):
    '''Returns (cams, pi, pred, logits) for a single image (3, 224, 224).
       Hook BEFORE block_idx so that block_idx's self-attention propagates
       gradient from CLS_out back to all patch tokens at the input of block_idx.
    '''
    img = img.unsqueeze(0).to(device)
    img.requires_grad_(True)
    M = probe.num_experts

    with torch.enable_grad():
        x = vit.patch_embed(img)
        x = vit._pos_embed(x)
        if hasattr(vit, "patch_drop"):
            x = vit.patch_drop(x)
        if hasattr(vit, "norm_pre"):
            x = vit.norm_pre(x)

        T = None
        cls_after = None
        for i, blk in enumerate(vit.blocks):
            if i == block_idx:
                T = x
                T.retain_grad()
            x = blk(x)
            if i == block_idx:
                if apply_final_norm and hasattr(vit, "norm"):
                    x_norm = vit.norm(x)
                    cls_after = x_norm[:, 0, :]
                else:
                    cls_after = x[:, 0, :]
                break
        if T is None or cls_after is None:
            raise ValueError(f"block_idx={block_idx} not reached in {len(vit.blocks)} blocks")

        h_stack = torch.stack([E(cls_after) for E in probe.experts], dim=1)
        pi = F.softmax(probe.router(cls_after), dim=-1)
        logits = probe.classifier((pi.unsqueeze(-1) * h_stack).sum(dim=1))
        pred = int(logits.argmax(dim=1).item())

        cams = []
        for m in range(M):
            if T.grad is not None:
                T.grad.zero_()
            contrib = pi[:, m:m + 1] * h_stack[:, m, :]
            target = probe.classifier(contrib)[0, pred]
            target.backward(retain_graph=(m < M - 1))

            grad_patches = T.grad[0, 1:, :]
            act_patches = T[0, 1:, :].detach()
            weight = grad_patches.mean(dim=0)
            cam = F.relu((act_patches * weight).sum(dim=-1))
            cam = cam.reshape(grid_size, grid_size)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams.append(cam.detach().cpu().numpy())

    pi_np = pi[0].detach().cpu().numpy()
    logits_np = logits[0].detach().cpu().numpy()
    return cams, pi_np, pred, logits_np


def upsample_cam(cam, size=224):
    t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def render_figure(img_uint, cams, pi, pred, true_y, group, block_idx, fmt, out_path):
    M = len(cams)
    fig, axes = plt.subplots(1, M + 1, figsize=(3 * (M + 1), 3.4))
    axes[0].imshow(img_uint)
    axes[0].set_title("input")
    axes[0].axis("off")
    for m in range(M):
        cam_up = upsample_cam(cams[m])
        axes[m + 1].imshow(img_uint)
        axes[m + 1].imshow(cam_up, cmap="jet", alpha=0.5)
        axes[m + 1].set_title(f"E{m}  π={pi[m]:.2f}")
        axes[m + 1].axis("off")
    fig.suptitle(
        f"[{fmt}] block={block_idx}  group={group}  true={true_y}  pred={pred}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-ckpt", required=True,
                    help="MoE ckpt (auto-detect: old probe / stage1 combined / stage2 MoE-only)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--backbone-checkpoint", default=None,
                    help="DeiT backbone source for old/stage2 ckpts; ignored for stage1")
    ap.add_argument("--n-per-group", type=int, default=2)
    ap.add_argument("--out-dir", default=None,
                    help="default: outputs/expert_heatmaps/<probe-stem>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit, probe, block_idx, apply_final_norm, fmt, cfg = load_components(
        args.probe_ckpt, args.backbone_checkpoint, device
    )
    backbone_name = cfg.get("backbone", "deit_*_patch16_224")
    embed_dim = cfg.get("embed_dim", cfg.get("input_size"))
    print(f"[fmt] {fmt}  backbone={backbone_name}  embed_dim={embed_dim}  "
          f"num_experts={probe.num_experts}  block_idx={block_idx}  "
          f"apply_final_norm={apply_final_norm}")

    grid = int(math.isqrt(vit.patch_embed.num_patches))
    if grid * grid != vit.patch_embed.num_patches:
        raise ValueError(f"non-square patch grid: {vit.patch_embed.num_patches}")
    print(f"[backbone] depth={len(vit.blocks)} patch_grid={grid}x{grid}")

    tfm = build_transforms()
    base = MyWaterBirdsDataset(args.data_root, remove_minority_groups=False, transform=tfm)
    samples = select_samples(base, args.n_per_group, args.seed)
    print(f"[samples] selected {len(samples)} images "
          f"({args.n_per_group} per group across {base.n_groups} groups)")

    out_dir = Path(args.out_dir or f"outputs/expert_heatmaps/{Path(args.probe_ckpt).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    print("\nidx  group  y  pred  pi=[{}]  expert-argmax(row,col)".format(
        " ".join(f"e{m}" for m in range(probe.num_experts))))
    for idx, group, y in samples:
        sample = base[idx]
        img_tensor = sample[0][0]
        cams, pi, pred, logits = expert_gradcam(
            vit, probe, img_tensor, block_idx, apply_final_norm, device, grid_size=grid
        )
        img_uint = denormalize(img_tensor)
        out_path = out_dir / f"idx{idx}_group{group}_y{y}_pred{pred}.png"
        render_figure(img_uint, cams, pi, pred, y, group, block_idx, fmt, out_path)

        argmax_str = " ".join(
            "E{}:({:>2},{:>2})".format(m, *np.unravel_index(int(np.argmax(c)), c.shape))
            for m, c in enumerate(cams)
        )
        pi_str = " ".join(f"{p:.2f}" for p in pi)
        print(f"{idx:>4}  {group:>4}  {y}   {pred}   [{pi_str}]  {argmax_str}")

    print(f"\nsaved {len(samples)} figure(s) to {out_dir}")


if __name__ == "__main__":
    main()
