''' Stage 2: continue-train the MoE head on a frozen DeiT backbone (CE + optional aux losses).

    Loads the combined ckpt produced by Stage 1 (finetune_deit_waterbirds.py),
    freezes the backbone, and keeps training only the MoE head — selects the best
    epoch by val worst-group accuracy and saves the refined MoE.

    Usage:
        PYTHONPATH=. python examples/CelebA-Waterbirds/deit_moe_zero_shot.py \
            --data-root ./data/waterbirds/waterbird_complete95_forest2water2 \
            --backbone-checkpoint ./checkpoints/deit_small_moe_waterbirds.pth \
            --output-path ./checkpoints/deit_small_moe_finetuned.pth
'''

import argparse
import math
import os
import random

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from intermediate_gen.datasets import MyWaterBirdsDataset
from intermediate_gen import MoEProbeModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def unpack(batch):
    (img, y, _attr, _idx), (group, _) = batch
    return img, torch.as_tensor(y, dtype=torch.long), torch.as_tensor(group, dtype=torch.long)


def collate(samples):
    imgs, ys, groups = zip(*[unpack(s) for s in samples])
    return torch.stack(imgs), torch.stack(ys), torch.stack(groups)


def load_stage1(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if not ("backbone_state" in ckpt and "moe_state" in ckpt and "config" in ckpt):
        raise ValueError(
            f"expected combined Stage-1 ckpt with keys "
            f"{{backbone_state, moe_state, config}}, got: {list(ckpt.keys())}"
        )
    return ckpt


def build_backbone(cfg, ckpt, device):
    backbone = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0)
    backbone.load_state_dict(ckpt["backbone_state"])
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def build_moe(cfg, ckpt, args, device):
    moe = MoEProbeModel(
        input_size=cfg["embed_dim"],
        output_size=cfg["num_classes"],
        num_experts=cfg["num_experts"],
        lr=args.lr,
        lambda_div=args.lambda_div,
        lambda_sp=args.lambda_sp,
        lambda_bal=args.lambda_bal,
        bal_beta=args.bal_beta,
        optimizer="adam",
    ).to(device)
    missing, unexpected = moe.load_state_dict(ckpt["moe_state"], strict=False)
    if unexpected:
        print(f"[moe] WARNING unexpected keys in ckpt: {unexpected}")
    if missing:
        bal_buffers = [k for k in missing if k.startswith("balance_loss_fn.")]
        other = [k for k in missing if not k.startswith("balance_loss_fn.")]
        if bal_buffers:
            print(f"[moe] balance_loss_fn buffers not in Stage-1 ckpt, initialised fresh: "
                  f"{bal_buffers}")
        if other:
            print(f"[moe] WARNING missing keys not from balance_loss_fn: {other}")
    return moe


@torch.no_grad()
def evaluate(backbone, moe, loader, device, n_groups):
    backbone.eval()
    moe.eval()
    g_correct = np.zeros(n_groups, dtype=np.int64)
    g_total = np.zeros(n_groups, dtype=np.int64)
    for x, y, g in loader:
        x = x.to(device)
        y = y.to(device)
        z = backbone(x)
        logits, _, _ = moe(z)
        preds = logits.argmax(1).cpu().numpy()
        ok = (preds == y.cpu().numpy())
        g_np = g.numpy()
        for gi in range(n_groups):
            mask = (g_np == gi)
            g_correct[gi] += int(ok[mask].sum())
            g_total[gi] += int(mask.sum())
    per_group = g_correct / np.maximum(g_total, 1)
    return float(per_group.mean()), float(per_group.min()), per_group


def save_moe_ckpt(moe, cfg, args, epoch, val_avg, val_worst, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        "state_dict": moe.state_dict(),
        "val_worst": val_worst,
        "val_avg": val_avg,
        "epoch": epoch,
        "config": {
            "input_size": cfg["embed_dim"],
            "output_size": cfg["num_classes"],
            "num_experts": cfg["num_experts"],
            "lambda_div": args.lambda_div,
            "lambda_sp": args.lambda_sp,
            "lambda_bal": args.lambda_bal,
            "bal_beta": args.bal_beta,
            "balance_sampler": args.balance_sampler,
            "backbone": cfg["backbone"],
            "stage1_ckpt": args.backbone_checkpoint,
        },
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="path to waterbird_complete95_forest2water2 (with metadata.csv)")
    ap.add_argument("--backbone-checkpoint", required=True,
                    help="combined Stage-1 ckpt (backbone+MoE) from finetune_deit_waterbirds.py")
    ap.add_argument("--output-path", default="./checkpoints/deit_small_moe_finetuned.pth",
                    help="path to save refined MoE state_dict (Stage 2 output)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lambda-div", type=float, default=0.02)
    ap.add_argument("--lambda-sp", type=float, default=0.02)
    ap.add_argument("--lambda-bal", type=float, default=0.02)
    ap.add_argument("--bal-beta", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--balance-sampler", choices=["weighted", "none"], default="weighted",
                    help="group-balanced sampling for Stage 2 train loader (default weighted)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = load_stage1(args.backbone_checkpoint, device)
    cfg = ckpt["config"]
    print(f"[stage1 ckpt] backbone={cfg['backbone']} embed_dim={cfg['embed_dim']} "
          f"num_experts={cfg['num_experts']} num_classes={cfg['num_classes']} "
          f"src_epoch={ckpt.get('epoch','?')} src_val_worst={ckpt.get('val_worst','?')}")

    tfm = build_transforms()
    base = MyWaterBirdsDataset(args.data_root, remove_minority_groups=False, transform=tfm)
    train_ds = Subset(base, base.train_idxs)
    val_ds = Subset(base, base.val_idxs)
    test_ds = Subset(base, base.test_idxs)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"num_classes={base.n_classes} n_groups={base.n_groups}")

    if args.balance_sampler == "weighted":
        train_group_arr = base.group_array[base.train_idxs]
        group_counts = np.bincount(train_group_arr, minlength=base.n_groups)
        sample_weights = 1.0 / group_counts[train_group_arr]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate)
        print(f"[balance] group counts={group_counts.tolist()} → weighted sampler "
              f"(per-group expected ≈ {1.0/base.n_groups:.3f})")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate)
        print(f"[balance] no sampler — using natural train distribution")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)

    backbone = build_backbone(cfg, ckpt, device)
    moe = build_moe(cfg, ckpt, args, device)
    print(f"lambda_div={args.lambda_div} lambda_sp={args.lambda_sp} "
          f"lambda_bal={args.lambda_bal} bal_beta={args.bal_beta} lr={args.lr}")
    print(f"out_path={args.output_path}")

    best_val_worst = -math.inf
    for epoch in range(args.epochs):
        moe.train()
        running = {"cls": 0.0, "div": 0.0, "sp": 0.0, "bal": 0.0, "acc": 0.0, "n": 0}
        for step, (x, y, _g) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                z = backbone(x)
            stats = moe.step_loss(z, y)
            running["cls"] += stats["loss_cls"]
            running["div"] += stats["loss_div"]
            running["sp"] += stats["loss_sp"]
            running["bal"] += stats["loss_bal"]
            running["acc"] += stats["accuracy"]
            running["n"] += 1
            if step % args.log_every == 0:
                n = max(running["n"], 1)
                print(f"[epoch {epoch} step {step}] "
                      f"loss_cls={running['cls']/n:.4f} loss_div={running['div']/n:.4f} "
                      f"loss_sp={running['sp']/n:.4f} loss_bal={running['bal']/n:.4f} "
                      f"acc={running['acc']/n:.4f}")

        val_avg, val_worst, per_group = evaluate(backbone, moe, val_loader, device, base.n_groups)
        pg_str = " ".join(f"{v:.3f}" for v in per_group)
        print(f"  [epoch {epoch}] VAL avg={val_avg:.4f} worst={val_worst:.4f} "
              f"per-group=[{pg_str}]")
        if val_worst > best_val_worst:
            best_val_worst = val_worst
            save_moe_ckpt(moe, cfg, args, epoch, val_avg, val_worst, args.output_path)
            print(f"      [save] new best -> {args.output_path}")

    best = torch.load(args.output_path, map_location=device)
    moe.load_state_dict(best["state_dict"])
    test_avg, test_worst, per_group = evaluate(backbone, moe, test_loader, device, base.n_groups)
    pg_str = " ".join(f"{v:.3f}" for v in per_group)
    print(f"\nfinal TEST avg={test_avg:.4f} worst={test_worst:.4f} per-group=[{pg_str}]")
    print(f"best epoch={best['epoch']} val_worst={best['val_worst']:.4f}")


if __name__ == "__main__":
    main()
