''' Stage 1: joint fine-tune DeiT backbone + MoE head on Waterbirds (CE-only).

    Saves a combined checkpoint with {backbone_state, moe_state, config} to be loaded
    by examples/CelebA-Waterbirds/deit_moe_zero_shot.py (Stage 2) for frozen-backbone
    MoE refinement.

    Usage:
        python examples/CelebA-Waterbirds/finetune_deit_waterbirds.py \
            --data-root ./data/waterbirds/waterbird_complete95_forest2water2 \
            --output-path ./checkpoints/deit_small_moe_waterbirds.pth
'''

import argparse
import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from intermediate_gen.datasets import MyWaterBirdsDataset
from intermediate_gen import MoEProbeModel, loss_balance, loss_div, loss_sparse


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_tfm():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_tfm():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def unpack(sample):
    (img, y, _attr, _idx), (group, _) = sample
    return img, torch.as_tensor(y, dtype=torch.long), torch.as_tensor(group, dtype=torch.long)


def collate(samples):
    imgs, ys, gs = zip(*[unpack(s) for s in samples])
    return torch.stack(imgs), torch.stack(ys), torch.stack(gs)


class MoEDeiT(nn.Module):
    '''DeiT backbone (timm, num_classes=0 → returns CLS post-norm) + MoE head.'''

    def __init__(self, backbone_name, num_classes, num_experts):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.embed_dim = self.backbone.num_features
        self.head = MoEProbeModel(
            input_size=self.embed_dim,
            output_size=num_classes,
            num_experts=num_experts,
            optimizer="adam",
        )

    def forward(self, x):
        z = self.backbone(x)
        logits, pi, h_stack = self.head(z)
        return logits, pi, h_stack


@torch.no_grad()
def eval_worst_group(model, loader, device, n_groups):
    model.eval()
    correct = torch.zeros(n_groups)
    total = torch.zeros(n_groups)
    for x, y, g in loader:
        x = x.to(device)
        logits, _, _ = model(x)
        preds = logits.argmax(1).cpu()
        ok = (preds == y)
        for gi in range(n_groups):
            mask = (g == gi)
            total[gi] += mask.sum().item()
            correct[gi] += (ok & mask).sum().item()
    per_group = correct / total.clamp_min(1)
    return per_group.mean().item(), per_group.min().item(), per_group.tolist()


def compute_stage1_loss(model, logits, pi, h_stack, y, args):
    loss_cls = F.cross_entropy(logits, y)
    l_div = loss_div(h_stack)
    l_sp = loss_sparse(pi)
    l_bal = loss_balance(pi)
    loss = (
        loss_cls
        + args.lambda_div * l_div
        + args.lambda_sp * l_sp
        + args.lambda_bal * l_bal
    )
    return loss, loss_cls, l_div, l_sp, l_bal


def save_combined(model, args, epoch, val_avg, val_worst, n_classes, path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        "backbone_state": model.backbone.state_dict(),
        "moe_state": model.head.state_dict(),
        "config": {
            "backbone": args.model,
            "embed_dim": model.embed_dim,
            "num_experts": args.num_experts,
            "num_classes": n_classes,
            "stage1_objective": "ce_plus_moe_aux",
            "lambda_div": args.lambda_div,
            "lambda_sp": args.lambda_sp,
            "lambda_bal": args.lambda_bal,
        },
        "epoch": epoch,
        "val_avg": val_avg,
        "val_worst": val_worst,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="path to waterbird_complete95_forest2water2 (with metadata.csv)")
    ap.add_argument("--model", default="deit_small_patch16_224")
    ap.add_argument("--output-path", default="./checkpoints/deit_small_moe_waterbirds.pth")
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lambda-div", type=float, default=0.02)
    ap.add_argument("--lambda-sp", type=float, default=0.02)
    ap.add_argument("--lambda-bal", type=float, default=0.02)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_base = MyWaterBirdsDataset(args.data_root, remove_minority_groups=False,
                                     transform=build_train_tfm())
    eval_base = MyWaterBirdsDataset(args.data_root, remove_minority_groups=False,
                                    transform=build_eval_tfm())
    train_loader = DataLoader(Subset(train_base, train_base.train_idxs),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(Subset(eval_base, eval_base.val_idxs),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)
    test_loader = DataLoader(Subset(eval_base, eval_base.test_idxs),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)} n_groups={train_base.n_groups}")

    model = MoEDeiT(args.model, num_classes=train_base.n_classes,
                    num_experts=args.num_experts).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"backbone={args.model} embed_dim={model.embed_dim} num_experts={args.num_experts} "
          f"epochs={args.epochs} lr={args.lr} wd={args.weight_decay} batch={args.batch_size} "
          f"lambda_div={args.lambda_div} lambda_sp={args.lambda_sp} "
          f"lambda_bal={args.lambda_bal}")

    best_val_worst = -1.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_div = 0.0
        running_sp = 0.0
        running_bal = 0.0
        running_acc = 0.0
        n = 0
        for step, (x, y, _g) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            logits, pi, h_stack = model(x)
            loss, loss_cls, l_div, l_sp, l_bal = compute_stage1_loss(
                model, logits, pi, h_stack, y, args
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            running_cls += loss_cls.item()
            running_div += l_div.item()
            running_sp += l_sp.item()
            running_bal += l_bal.item()
            running_acc += (logits.argmax(1) == y).float().mean().item()
            n += 1
            if step % 20 == 0:
                print(f"[epoch {epoch} step {step}] loss={running_loss/n:.4f} "
                      f"loss_cls={running_cls/n:.4f} loss_div={running_div/n:.4f} "
                      f"loss_sp={running_sp/n:.4f} loss_bal={running_bal/n:.4f} "
                      f"acc={running_acc/n:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        sched.step()

        avg_v, worst_v, pg_v = eval_worst_group(model, val_loader, device, train_base.n_groups)
        pg_str = " ".join(f"{v:.3f}" for v in pg_v)
        print(f"  [epoch {epoch}] VAL avg={avg_v:.4f} worst={worst_v:.4f} per-group=[{pg_str}]")
        if worst_v > best_val_worst:
            best_val_worst = worst_v
            save_combined(model, args, epoch, avg_v, worst_v,
                          train_base.n_classes, args.output_path)
            print(f"  [epoch {epoch}] saved (best val worst={worst_v:.4f}) -> {args.output_path}")

    avg_t, worst_t, pg_t = eval_worst_group(model, test_loader, device, train_base.n_groups)
    pg_str = " ".join(f"{v:.3f}" for v in pg_t)
    print(f"\nfinal TEST avg={avg_t:.4f} worst={worst_t:.4f} per-group=[{pg_str}]")
    print(f"best val worst-group acc = {best_val_worst:.4f}; checkpoint at {args.output_path}")


if __name__ == "__main__":
    main()
