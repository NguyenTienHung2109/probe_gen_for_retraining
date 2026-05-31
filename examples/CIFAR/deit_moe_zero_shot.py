''' Zero-shot intermediate-layer MoE probing with a frozen DeiT backbone on CIFAR-10.

    Trains one MoEProbeModel per transformer block on CIFAR-10 train, then reports
    per-block accuracy on CIFAR-10 test.

    Usage:
        python examples/CIFAR/deit_moe_zero_shot.py --cifar-root /path/to/cifar-10-batches-py
'''

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from intermediate_gen.datasets import CIFARDataset
from intermediate_gen import IntermediateDEIT, MoEProbeModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def evaluate(deit, probes, loader, device):
    correct = {i: 0 for i in probes}
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long()
        total += y.size(0)
        for i, h_cls in deit.forward_intermediates(x):
            if i not in probes:
                continue
            logits = probes[i](h_cls)[0]
            correct[i] += (logits.argmax(dim=1) == y).sum().item()
    return {i: correct[i] / total for i in probes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar-root", required=True,
                    help="path to cifar-10-batches-py (with data_batch_1..5 and test_batch)")
    ap.add_argument("--model", default="deit_small_patch16_224")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--lambda-div", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tfm = build_transforms()
    train_ds = CIFARDataset(args.cifar_root, train=True, transform=tfm)
    test_ds = CIFARDataset(args.cifar_root, train=False, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    deit = IntermediateDEIT(args.model, pretrained=True).to(device)
    probes = {
        i: MoEProbeModel(
            input_size=deit.embed_dim,
            output_size=10,
            num_experts=args.num_experts,
            lr=args.lr,
            lambda_div=args.lambda_div,
            optimizer="adam",
        ).to(device)
        for i in range(deit.depth)
    }
    print(f"backbone={args.model} depth={deit.depth} embed_dim={deit.embed_dim} "
          f"num_experts={args.num_experts} lambda_div={args.lambda_div}")

    for epoch in range(args.epochs):
        running = {i: {"cls": 0.0, "div": 0.0, "acc": 0.0, "n": 0} for i in probes}
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device).long()
            for i, h_cls in deit.forward_intermediates(x):
                stats = probes[i].step_loss(h_cls.detach(), y)
                r = running[i]
                r["cls"] += stats["loss_cls"]
                r["div"] += stats["loss_div"]
                r["acc"] += stats["accuracy"]
                r["n"] += 1
            if step % 50 == 0:
                last = deit.depth - 1
                r = running[last]
                print(f"[epoch {epoch} step {step}] last-block "
                      f"loss_cls={r['cls']/max(r['n'],1):.4f} "
                      f"loss_div={r['div']/max(r['n'],1):.4f} "
                      f"acc={r['acc']/max(r['n'],1):.4f}")

    accs = evaluate(deit, probes, test_loader, device)
    print("\nper-block CIFAR-10 test accuracy:")
    for i in sorted(accs):
        print(f"  block {i:2d}: {accs[i]:.4f}")


if __name__ == "__main__":
    main()
