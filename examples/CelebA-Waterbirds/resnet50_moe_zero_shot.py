''' ResNet-50 + MoE intermediate-layer probing on Waterbirds.

    Trains one MoEProbeModel per probed ResNet stage on the Waterbirds train split,
    selects the best probe per layer by validation worst-group accuracy, saves each
    best probe to disk, then reports per-layer test accuracy using the loaded best.

    Backbone defaults to the DFR ResNet-50 checkpoint from Kirichenko et al. 2023
    (auto-downloaded to ./models/ on first run).

    Usage:
        PYTHONPATH=. python examples/CelebA-Waterbirds/resnet50_moe_zero_shot.py \
            --data-root ./data/waterbirds/waterbird_complete95_forest2water2
'''

import argparse
import math
import os
import random

import gdown
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50

from intermediate_gen.datasets import MyWaterBirdsDataset
from intermediate_gen import MoEProbeModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LAYERS_TO_PROBE = (3, 4, 5, 6, 7, 8)
DFR_GDRIVE_ID = "1gZDj5oIEJZOo9WgPgyrELOCE_C_hh9vA"
DFR_DEFAULT_PATH = "models/waterbirds_last_layer_retrained.pth"


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


def unpack(sample):
    (img, y, _attr, _idx), (group, _) = sample
    return (
        img,
        torch.as_tensor(y, dtype=torch.long),
        torch.as_tensor(group, dtype=torch.long),
    )


def collate(samples):
    imgs, ys, groups = zip(*[unpack(s) for s in samples])
    return torch.stack(imgs), torch.stack(ys), torch.stack(groups)


def ensure_backbone_ckpt(path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={DFR_GDRIVE_ID}"
    print(f"[backbone] downloading DFR ResNet-50 ckpt to {path}")
    gdown.download(url, path, quiet=False)
    return path


def load_resnet50_dfr(ckpt_path, device):
    model = resnet50(weights=None, num_classes=2)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(device)


def iter_features(model, x):
    '''Forward x through model.children() sequentially.
       Yield (layer_idx, pooled_feat) for layer_idx in LAYERS_TO_PROBE.
       pooled_feat = adaptive_avg_pool2d(act, 1).flatten(1) -> (B, C).
    '''
    for layer_idx, layer in enumerate(model.children()):
        x = layer(x if layer_idx <= 8 else x.view(x.size(0), -1))
        if layer_idx in LAYERS_TO_PROBE:
            if x.dim() == 4:
                h = F.adaptive_avg_pool2d(x, 1).flatten(1)
            else:
                h = x.flatten(1)
            yield layer_idx, h


@torch.no_grad()
def probe_channel_dims(model, device):
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    return {i: h.shape[1] for i, h in iter_features(model, dummy)}


def build_probes(channel_dims, n_classes, args, device):
    probes = {}
    for i, c in channel_dims.items():
        probes[i] = MoEProbeModel(
            input_size=c,
            output_size=n_classes,
            num_experts=args.num_experts,
            lr=args.lr,
            lambda_div=args.lambda_div,
            lambda_sp=args.lambda_sp,
            lambda_bal=args.lambda_bal,
            bal_beta=args.bal_beta,
            optimizer="adam",
        ).to(device)
    return probes


@torch.no_grad()
def evaluate(model, probes, loader, device, n_groups):
    g_correct = {i: np.zeros(n_groups, dtype=np.int64) for i in probes}
    g_total = np.zeros(n_groups, dtype=np.int64)
    for x, y, g in loader:
        x = x.to(device)
        y = y.to(device)
        g_np = g.numpy()
        for gi in range(n_groups):
            g_total[gi] += int((g_np == gi).sum())
        for i, h in iter_features(model, x):
            if i not in probes:
                continue
            preds = probes[i](h)[0].argmax(dim=1).cpu().numpy()
            ok = (preds == y.cpu().numpy())
            for gi in range(n_groups):
                g_correct[i][gi] += int(ok[g_np == gi].sum())
    out = {}
    for i in probes:
        per_group = g_correct[i] / np.maximum(g_total, 1)
        out[i] = {
            "avg": float(per_group.mean()),
            "worst": float(per_group.min()),
            "per_group": per_group,
        }
    return out


def save_probe_ckpt(probe, layer_idx, val_wga, val_acc, epoch, args, input_size, n_classes, out_dir):
    payload = {
        "state_dict": probe.state_dict(),
        "layer_idx": layer_idx,
        "val_wga": val_wga,
        "val_acc": val_acc,
        "epoch": epoch,
        "config": {
            "input_size": input_size,
            "output_size": n_classes,
            "num_experts": args.num_experts,
            "lambda_div": args.lambda_div,
            "lambda_sp": args.lambda_sp,
            "lambda_bal": args.lambda_bal,
            "bal_beta": args.bal_beta,
        },
    }
    path = os.path.join(out_dir, f"probe_layer{layer_idx}.pth")
    torch.save(payload, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="path to waterbird_complete95_forest2water2 (with metadata.csv)")
    ap.add_argument("--backbone-ckpt-path", default=DFR_DEFAULT_PATH,
                    help="path to ResNet-50 ckpt (auto-downloads DFR weights if missing)")
    ap.add_argument("--out-dir", default="checkpoints/r50_moe_probes")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--lambda-div", type=float, default=0.02)
    ap.add_argument("--lambda-sp", type=float, default=0.02)
    ap.add_argument("--lambda-bal", type=float, default=0.02)
    ap.add_argument("--bal-beta", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = build_transforms()
    base = MyWaterBirdsDataset(args.data_root, remove_minority_groups=False, transform=tfm)
    train_ds = Subset(base, base.train_idxs)
    val_ds = Subset(base, base.val_idxs)
    test_ds = Subset(base, base.test_idxs)
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"num_classes={base.n_classes} n_groups={base.n_groups}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)

    backbone = load_resnet50_dfr(ensure_backbone_ckpt(args.backbone_ckpt_path), device)
    print(f"[backbone] loaded ResNet-50 from {args.backbone_ckpt_path} (frozen, eval)")

    channel_dims = probe_channel_dims(backbone, device)
    print("probed layers and pooled channel dims:")
    for i, c in channel_dims.items():
        print(f"  layer {i}: C={c}")

    probes = build_probes(channel_dims, base.n_classes, args, device)
    print(f"num_experts={args.num_experts} lambda_div={args.lambda_div} "
          f"lambda_sp={args.lambda_sp} lambda_bal={args.lambda_bal} bal_beta={args.bal_beta} "
          f"lr={args.lr} epochs={args.epochs} batch={args.batch_size}")

    os.makedirs(args.out_dir, exist_ok=True)
    best = {i: {"wga": -math.inf, "acc": 0.0, "epoch": -1} for i in probes}

    for epoch in range(args.epochs):
        running = {i: {"cls": 0.0, "div": 0.0, "sp": 0.0, "bal": 0.0, "acc": 0.0, "n": 0}
                   for i in probes}
        for step, (x, y, _g) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            for i, h in iter_features(backbone, x):
                stats = probes[i].step_loss(h.detach(), y)
                r = running[i]
                r["cls"] += stats["loss_cls"]
                r["div"] += stats["loss_div"]
                r["sp"] += stats["loss_sp"]
                r["bal"] += stats["loss_bal"]
                r["acc"] += stats["accuracy"]
                r["n"] += 1
            if step % args.log_every == 0:
                last = max(probes)
                r = running[last]
                n = max(r["n"], 1)
                print(f"[epoch {epoch} step {step}] last-block "
                      f"loss_cls={r['cls']/n:.4f} loss_div={r['div']/n:.4f} "
                      f"loss_sp={r['sp']/n:.4f} loss_bal={r['bal']/n:.4f} "
                      f"acc={r['acc']/n:.4f}")

        val_results = evaluate(backbone, probes, val_loader, device, base.n_groups)
        print(f"  [epoch {epoch}] val per-layer worst-group:")
        for i in sorted(val_results):
            r = val_results[i]
            print(f"    layer {i}: avg={r['avg']:.4f} worst={r['worst']:.4f}")
            if r["worst"] > best[i]["wga"]:
                best[i] = {"wga": r["worst"], "acc": r["avg"], "epoch": epoch}
                path = save_probe_ckpt(
                    probes[i], i, r["worst"], r["avg"], epoch, args,
                    input_size=channel_dims[i], n_classes=base.n_classes,
                    out_dir=args.out_dir,
                )
                print(f"      [save] new best -> {path}")

    print("\nbest validation worst-group per layer:")
    for i in sorted(best):
        b = best[i]
        print(f"  layer {i}: val_wga={b['wga']:.4f} val_acc={b['acc']:.4f} epoch={b['epoch']}")

    for i in probes:
        path = os.path.join(args.out_dir, f"probe_layer{i}.pth")
        ckpt = torch.load(path, map_location=device)
        probes[i].load_state_dict(ckpt["state_dict"])

    test_results = evaluate(backbone, probes, test_loader, device, base.n_groups)
    print("\nper-layer Waterbirds TEST accuracy (using best-by-val-WGA probe):")
    print(f"  {'layer':>6}  {'avg':>7}  {'worst':>7}  per-group")
    for i in sorted(test_results):
        r = test_results[i]
        pg = " ".join(f"{v:.3f}" for v in r["per_group"])
        print(f"  {i:>6}  {r['avg']:.4f}  {r['worst']:.4f}  [{pg}]")


if __name__ == "__main__":
    main()
