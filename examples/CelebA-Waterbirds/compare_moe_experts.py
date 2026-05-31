''' Side-by-side Grad-CAM comparison: Stage 1 vs Stage 2 MoE on shared images.

    Filters test samples where Stage 1 predicts WRONG and Stage 2 predicts RIGHT,
    then renders one 2-row figure per sample (top = Stage 1 heatmaps, bottom =
    Stage 2 heatmaps) so you can see what Stage 2 refinement fixed.

    Usage:
        PYTHONPATH=. python examples/CelebA-Waterbirds/compare_moe_experts.py \
            --stage1-ckpt checkpoints/deit_small_moe_waterbirds.pth \
            --stage2-ckpt checkpoints/deit_small_moe_finetuned.pth \
            --data-root ./data/waterbirds/waterbird_complete95_forest2water2 \
            --max-samples 8 \
            --out-dir outputs/expert_heatmaps/compare_s1wrong_s2right
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
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

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
    elif fmt == "stage2":
        backbone_name = cfg["backbone"]
        bb_path = backbone_ckpt_path or cfg.get("stage1_ckpt")
        if not bb_path or not Path(bb_path).exists():
            raise ValueError(
                f"Stage-2 ckpt needs --backbone-checkpoint or valid config['stage1_ckpt']; "
                f"got {cfg.get('stage1_ckpt')!r}"
            )
        bb_ckpt = torch.load(bb_path, map_location=device)
        if "backbone_state" not in bb_ckpt:
            raise ValueError(f"backbone source {bb_path} is not a Stage-1 ckpt")
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
    else:
        raise ValueError(f"this script only supports stage1/stage2 ckpts, not {fmt!r}")

    vit = timm.create_model(backbone_name, pretrained=False, num_classes=0)
    missing, unexpected = vit.load_state_dict(backbone_state, strict=False)
    bad_missing = [k for k in missing if not k.startswith("head.")]
    if bad_missing:
        print(f"[backbone {fmt}] WARNING missing keys: {bad_missing}")
    rest = [k for k in unexpected if not k.startswith("head.")]
    if rest:
        print(f"[backbone {fmt}] WARNING unexpected keys: {rest}")
    vit.eval().to(device)
    for p in vit.parameters():
        p.requires_grad_(False)

    if block_idx is None:
        block_idx = len(vit.blocks) - 1

    moe = MoEProbeModel(**moe_kwargs)
    moe.load_state_dict(moe_state, strict=False)
    moe.eval().to(device)
    for p in moe.parameters():
        p.requires_grad_(False)

    return vit, moe, block_idx, apply_final_norm, fmt, cfg


@torch.no_grad()
def predict_class(vit, moe, img, block_idx, apply_final_norm, device):
    img = img.unsqueeze(0).to(device)
    x = vit.patch_embed(img)
    x = vit._pos_embed(x)
    if hasattr(vit, "patch_drop"):
        x = vit.patch_drop(x)
    if hasattr(vit, "norm_pre"):
        x = vit.norm_pre(x)
    for i, blk in enumerate(vit.blocks):
        x = blk(x)
        if i == block_idx:
            if apply_final_norm and hasattr(vit, "norm"):
                cls = vit.norm(x)[:, 0, :]
            else:
                cls = x[:, 0, :]
            break
    logits, _, _ = moe(cls)
    return int(logits.argmax(dim=1).item())


def expert_gradcam(vit, moe, img, block_idx, apply_final_norm, device, grid_size):
    img = img.unsqueeze(0).to(device)
    img.requires_grad_(True)
    M = moe.num_experts

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
                    cls_after = vit.norm(x)[:, 0, :]
                else:
                    cls_after = x[:, 0, :]
                break
        if T is None or cls_after is None:
            raise ValueError(f"block_idx={block_idx} not reached")

        h_stack = torch.stack([E(cls_after) for E in moe.experts], dim=1)
        pi = F.softmax(moe.router(cls_after), dim=-1)
        logits = moe.classifier((pi.unsqueeze(-1) * h_stack).sum(dim=1))
        pred = int(logits.argmax(dim=1).item())

        cams = []
        for m in range(M):
            if T.grad is not None:
                T.grad.zero_()
            contrib = pi[:, m:m + 1] * h_stack[:, m, :]
            target = moe.classifier(contrib)[0, pred]
            target.backward(retain_graph=(m < M - 1))

            grad_patches = T.grad[0, 1:, :]
            act_patches = T[0, 1:, :].detach()
            weight = grad_patches.mean(dim=0)
            cam = F.relu((act_patches * weight).sum(dim=-1))
            cam = cam.reshape(grid_size, grid_size)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cams.append(cam.detach().cpu().numpy())

    pi_np = pi[0].detach().cpu().numpy()
    return cams, pi_np, pred


def upsample_cam(cam, size=224):
    t = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def render_compare(img_uint, cams_s1, pi_s1, cams_s2, pi_s2,
                   y, g, p1, p2, blk_s1, blk_s2, out_path):
    M1, M2 = len(cams_s1), len(cams_s2)
    M = max(M1, M2)
    fig, axes = plt.subplots(2, M + 1, figsize=(3 * (M + 1), 6.6))

    for row, (cams, pi, label, blk, M_row) in enumerate([
        (cams_s1, pi_s1, "stage1 (WRONG)", blk_s1, M1),
        (cams_s2, pi_s2, "stage2 (RIGHT)", blk_s2, M2),
    ]):
        axes[row, 0].imshow(img_uint)
        axes[row, 0].set_title(f"{label}\ninput  block={blk}", fontsize=9)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for m in range(M):
            ax = axes[row, m + 1]
            if m < M_row:
                ax.imshow(img_uint)
                ax.imshow(upsample_cam(cams[m]), cmap="jet", alpha=0.5)
                ax.set_title(f"E{m}  π={pi[m]:.2f}")
            ax.axis("off")
    fig.suptitle(
        f"group={g}  true={y}  stage1_pred={p1}  stage2_pred={p2}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-ckpt", required=True,
                    help="Stage 1 combined ckpt (backbone+MoE)")
    ap.add_argument("--stage2-ckpt", required=True,
                    help="Stage 2 MoE-only ckpt (refined on frozen backbone)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--backbone-checkpoint", default=None,
                    help="override Stage-1 backbone source for Stage-2 (defaults to "
                         "Stage-2 config['stage1_ckpt'])")
    ap.add_argument("--max-samples", type=int, default=8,
                    help="cap number of (s1-wrong, s2-right) samples to visualize")
    ap.add_argument("--out-dir", default="outputs/expert_heatmaps/compare_s1wrong_s2right")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[stage1] loading...")
    vit_s1, moe_s1, blk_s1, norm_s1, fmt_s1, cfg_s1 = load_components(
        args.stage1_ckpt, None, device
    )
    print(f"  fmt={fmt_s1} backbone={cfg_s1['backbone']} block={blk_s1} "
          f"num_experts={moe_s1.num_experts}")

    print("[stage2] loading...")
    vit_s2, moe_s2, blk_s2, norm_s2, fmt_s2, cfg_s2 = load_components(
        args.stage2_ckpt, args.backbone_checkpoint or args.stage1_ckpt, device
    )
    print(f"  fmt={fmt_s2} backbone={cfg_s2['backbone']} block={blk_s2} "
          f"num_experts={moe_s2.num_experts}")

    if cfg_s1["backbone"] != cfg_s2["backbone"]:
        print(f"[warn] backbone mismatch: stage1={cfg_s1['backbone']} "
              f"stage2={cfg_s2['backbone']}")

    grid = int(math.isqrt(vit_s1.patch_embed.num_patches))

    tfm = build_transforms()
    base = MyWaterBirdsDataset(args.data_root, remove_minority_groups=False, transform=tfm)
    print(f"[data] test split has {len(base.test_idxs)} samples, n_groups={base.n_groups}")

    print("\n[scan] finding samples where stage1=WRONG and stage2=RIGHT...")
    selected = []
    for idx in tqdm(base.test_idxs):
        sample = base[int(idx)]
        img = sample[0][0]
        y = int(base.y_array[int(idx)])
        g = int(base.group_array[int(idx)])
        p1 = predict_class(vit_s1, moe_s1, img, blk_s1, norm_s1, device)
        p2 = predict_class(vit_s2, moe_s2, img, blk_s2, norm_s2, device)
        if p1 != y and p2 == y:
            selected.append((int(idx), g, y, p1, p2))
            if len(selected) >= args.max_samples:
                break
    print(f"[filter] {len(selected)} samples matched (capped at {args.max_samples})")

    if not selected:
        print("no matching samples — Stage 2 didn't fix any case Stage 1 missed.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}\n")

    print("idx  group  y  s1_pred  s2_pred  pi_s1                 pi_s2")
    for idx, g, y, p1, p2 in selected:
        sample = base[idx]
        img = sample[0][0]
        cams_s1, pi_s1, _ = expert_gradcam(
            vit_s1, moe_s1, img, blk_s1, norm_s1, device, grid_size=grid
        )
        cams_s2, pi_s2, _ = expert_gradcam(
            vit_s2, moe_s2, img, blk_s2, norm_s2, device, grid_size=grid
        )
        img_uint = denormalize(img)
        out_path = out_dir / f"idx{idx}_group{g}_y{y}_s1{p1}_s2{p2}.png"
        render_compare(img_uint, cams_s1, pi_s1, cams_s2, pi_s2,
                       y, g, p1, p2, blk_s1, blk_s2, out_path)
        s1_str = " ".join(f"{p:.2f}" for p in pi_s1)
        s2_str = " ".join(f"{p:.2f}" for p in pi_s2)
        print(f"{idx:>4}  {g:>4}  {y}    {p1}        {p2}        [{s1_str}]   [{s2_str}]")

    print(f"\nsaved {len(selected)} comparison figure(s) to {out_dir}")


if __name__ == "__main__":
    main()
