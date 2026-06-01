'''Phase 2: retrain only the final ResNet-50 layer on a sparse Waterbirds probe.

Usage:
    PYTHONPATH=. python examples/CelebA-Waterbirds/finetune_last_layer_with_sparse_probe.py \
        --probe-json probes/probe_oracle_group_balanced.json \
        --pretrained-checkpoint checkpoints/final_checkpoint.pt \
        --output-path checkpoints/resnet50_waterbirds_oracle_probe_last_layer.pth
'''

import argparse
import copy
import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet50

from intermediate_gen.datasets import MyWaterBirdsDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def unpack(sample):
    (img, y, attr, idx), (group, _) = sample
    return (
        img,
        torch.as_tensor(y, dtype=torch.long),
        torch.as_tensor(attr, dtype=torch.long),
        torch.as_tensor(group, dtype=torch.long),
        torch.as_tensor(idx, dtype=torch.long),
    )


def collate(samples):
    imgs, ys, attrs, groups, idxs = zip(*[unpack(s) for s in samples])
    return (
        torch.stack(imgs),
        torch.stack(ys),
        torch.stack(attrs),
        torch.stack(groups),
        torch.stack(idxs),
    )


def load_resnet50_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing checkpoint: {path}")

    model = resnet50(pretrained=False, num_classes=2)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                print(f"[checkpoint] using key: {key}")
                break

    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt):
        ckpt = {k.removeprefix("module."): v for k, v in ckpt.items()}

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"[checkpoint] loaded {path}")
    print(f"[checkpoint] missing keys: {missing}")
    print(f"[checkpoint] unexpected keys: {unexpected}")

    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.fc.parameters():
        p.requires_grad_(True)
    return model


def load_probe_indices(path, base_dataset):
    with open(path, "r", encoding="utf-8") as f:
        probe = json.load(f)

    selected_indices = [int(i) for i in probe["selected_indices"]]
    train_index_set = set(int(i) for i in base_dataset.train_idxs)
    selected_set = set(selected_indices)
    if not selected_set.issubset(train_index_set):
        bad = sorted(selected_set - train_index_set)[:10]
        raise ValueError(f"probe contains indices outside train split, examples={bad}")
    return probe, selected_indices


def summarize_probe(path, probe, selected_indices, base_dataset):
    selected_y = base_dataset.y_array[selected_indices]
    selected_p = base_dataset.p_array[selected_indices]
    selected_g = base_dataset.group_array[selected_indices]
    print(f"[probe] path: {path}")
    print(f"[probe] selected: {len(selected_indices)}")
    print(f"[probe] class counts: {dict(Counter(selected_y.tolist()))}")
    print(f"[probe] group counts: {dict(Counter(selected_g.tolist()))}")
    print(f"[probe] conflict fraction: {float(np.mean(selected_p != selected_y)):.4f}")
    if "hparams" in probe:
        print(f"[probe] hparams: {probe['hparams']}")


@torch.no_grad()
def evaluate(model, loader, device, n_groups):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    g_correct = np.zeros(n_groups, dtype=np.int64)
    g_total = np.zeros(n_groups, dtype=np.int64)

    for x, y, _attrs, groups, _idxs in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        preds = logits.argmax(dim=1)
        ok = preds.eq(y)

        total_loss += float(losses.sum().item())
        total += int(y.numel())
        correct += int(ok.sum().item())

        groups_np = groups.numpy()
        ok_np = ok.cpu().numpy()
        for g in range(n_groups):
            mask = groups_np == g
            g_total[g] += int(mask.sum())
            g_correct[g] += int(ok_np[mask].sum())

    per_group = g_correct / np.maximum(g_total, 1)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "worst_group_accuracy": float(per_group.min()),
        "avg_group_accuracy": float(per_group.mean()),
        "per_group_accuracy": per_group.tolist(),
        "per_group_total": g_total.tolist(),
    }


def save_checkpoint(path, model, epoch, val_metrics, selected_indices, args):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "fc_state": model.fc.state_dict(),
        "epoch": epoch,
        "val_metrics": val_metrics,
        "probe_json": args.probe_json,
        "selected_indices": selected_indices,
        "hparams": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "trainable": "fc_only",
            "pretrained_checkpoint": args.pretrained_checkpoint,
        },
    }, path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="./data/waterbirds/waterbird_complete95_forest2water2",
        help="path to waterbird_complete95_forest2water2 with metadata.csv",
    )
    ap.add_argument(
        "--probe-json",
        default="probes/probe_oracle_group_balanced.json",
        help="probe JSON containing selected_indices",
    )
    ap.add_argument(
        "--pretrained-checkpoint",
        default="checkpoints/final_checkpoint.pt",
        help="ResNet-50 checkpoint to load before last-layer retraining",
    )
    ap.add_argument(
        "--output-path",
        default="checkpoints/resnet50_waterbirds_oracle_probe_last_layer.pth",
    )
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--device", default=None, help="cuda, cpu, or unset for auto")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[device] gpu: {torch.cuda.get_device_name(device)}")

    base = MyWaterBirdsDataset(
        args.data_root, remove_minority_groups=False, transform=build_transform()
    )
    train_ds = Subset(base, base.train_idxs)
    val_ds = Subset(base, base.val_idxs)
    test_ds = Subset(base, base.test_idxs)
    print(f"[dataset] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        groups = base.group_array[ds.indices]
        print(f"[dataset] {name} group counts={np.bincount(groups, minlength=base.n_groups).tolist()}")

    probe, selected_indices = load_probe_indices(args.probe_json, base)
    summarize_probe(args.probe_json, probe, selected_indices, base)
    selected_ds = Subset(base, selected_indices)

    probe_loader = DataLoader(
        selected_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    model = load_resnet50_checkpoint(args.pretrained_checkpoint, device)
    print("[trainable]")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name}: {tuple(p.shape)}")

    before_val = evaluate(model, val_loader, device, base.n_groups)
    before_test = evaluate(model, test_loader, device, base.n_groups)
    print(f"[before] VAL {before_val}")
    print(f"[before] TEST {before_test}")

    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_state = copy.deepcopy(model.fc.state_dict())
    best_val_wga = -np.inf
    history = []

    for epoch in range(args.epochs):
        # Keep frozen backbone and BatchNorm statistics fixed; train only fc.
        model.eval()
        model.fc.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y, _attrs, _groups, _idxs in probe_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(y.numel())
            running_correct += int(logits.argmax(1).eq(y).sum().item())
            running_total += int(y.numel())

        val_metrics = evaluate(model, val_loader, device, base.n_groups)
        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_metrics": val_metrics,
        })

        if val_metrics["worst_group_accuracy"] > best_val_wga:
            best_val_wga = val_metrics["worst_group_accuracy"]
            best_state = copy.deepcopy(model.fc.state_dict())
            save_checkpoint(args.output_path, model, epoch, val_metrics, selected_indices, args)
            saved = " saved"
        else:
            saved = ""

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            pg = " ".join(f"{x:.3f}" for x in val_metrics["per_group_accuracy"])
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={val_metrics['accuracy']:.4f} "
                f"val_wga={val_metrics['worst_group_accuracy']:.4f} "
                f"per_group=[{pg}]{saved}"
            )

    model.fc.load_state_dict(best_state)
    final_val = evaluate(model, val_loader, device, base.n_groups)
    final_test = evaluate(model, test_loader, device, base.n_groups)
    print(f"[best] val_wga={best_val_wga:.4f}")
    print(f"[final] VAL {final_val}")
    print(f"[final] TEST {final_test}")
    print(f"[output] {args.output_path}")

    history_path = os.path.splitext(args.output_path)[0] + "_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)
    print(f"[history] {history_path}")


if __name__ == "__main__":
    main()
