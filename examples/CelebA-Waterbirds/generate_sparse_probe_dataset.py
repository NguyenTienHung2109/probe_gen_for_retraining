'''Generate a sparse expert-disagreement probe set for DeiT-MoE Waterbirds.

Usage:
    PYTHONPATH=. python examples/CelebA-Waterbirds/generate_sparse_probe_dataset.py \
        --data-root ./data/waterbirds/waterbird_complete95_forest2water2 \
        --checkpoint ./checkpoints/deit_small_moe_waterbirds.pth
'''

import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from intermediate_gen import MoEProbeModel
from intermediate_gen.datasets import MyWaterBirdsDataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EPS = 1e-8


@dataclass
class SelectionHparams:
    probe_percentage: float
    selection_method: str
    score_normalization: str
    alpha_pred: float
    beta_margin: float
    gamma_loss: float
    delta_router: float
    class_balance: str
    use_coverage: bool
    coverage_weight: float
    coverage_similarity: str
    coverage_universe: str
    similarity_mode: str
    use_redundancy: bool
    redundancy_weight: float
    candidate_pool_multiplier: int
    use_group_labels_for_selection: bool
    use_group_labels_for_eval: bool
    seed: int


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
    (img, y, attr, idx), (group, _) = sample
    place = int(attr[1])
    return (
        img,
        torch.as_tensor(y, dtype=torch.long),
        torch.as_tensor(group, dtype=torch.long),
        torch.as_tensor(place, dtype=torch.long),
        torch.as_tensor(idx, dtype=torch.long),
    )


def collate(samples):
    imgs, ys, groups, places, idxs = zip(*[unpack(s) for s in samples])
    return (
        torch.stack(imgs),
        torch.stack(ys),
        torch.stack(groups),
        torch.stack(places),
        torch.stack(idxs),
    )


def load_stage1(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    expected = {"backbone_state", "moe_state", "config"}
    if not expected.issubset(ckpt):
        raise ValueError(
            f"expected combined Stage-1 checkpoint with keys {sorted(expected)}, "
            f"got: {list(ckpt.keys())}"
        )
    return ckpt


def build_backbone(cfg, ckpt, device):
    backbone = timm.create_model(cfg["backbone"], pretrained=False, num_classes=0)
    backbone.load_state_dict(ckpt["backbone_state"])
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def build_moe(cfg, ckpt, device):
    moe = MoEProbeModel(
        input_size=cfg["embed_dim"],
        output_size=cfg["num_classes"],
        num_experts=cfg["num_experts"],
        optimizer="adam",
    ).to(device)
    missing, unexpected = moe.load_state_dict(ckpt["moe_state"], strict=False)
    if unexpected:
        print(f"[moe] WARNING unexpected keys in checkpoint: {unexpected}")
    other_missing = [
        k for k in missing if not k.startswith("balance_loss_fn.")
    ]
    if other_missing:
        print(f"[moe] WARNING missing keys not from balance buffers: {other_missing}")
    moe.eval()
    for p in moe.parameters():
        p.requires_grad_(False)
    return moe


def validate_model_config(cfg, backbone, moe, base):
    if cfg["num_classes"] != base.n_classes:
        raise ValueError(
            f"checkpoint num_classes={cfg['num_classes']} but dataset has "
            f"n_classes={base.n_classes}"
        )
    if cfg["num_experts"] != len(moe.experts):
        raise ValueError(
            f"checkpoint num_experts={cfg['num_experts']} but MoE has "
            f"{len(moe.experts)} experts"
        )
    if cfg["embed_dim"] != backbone.num_features:
        raise ValueError(
            f"checkpoint embed_dim={cfg['embed_dim']} but backbone.num_features="
            f"{backbone.num_features}"
        )


def expert_logits_and_probs(moe, h_stack):
    batch_size, num_experts, hidden_dim = h_stack.shape
    flat_h = h_stack.reshape(batch_size * num_experts, hidden_dim)
    flat_logits = moe.classifier(flat_h)
    logits = flat_logits.reshape(batch_size, num_experts, -1)
    probs = F.softmax(logits, dim=-1)
    return logits, probs


def score_batch(backbone, moe, x, y, device):
    x = x.to(device)
    y = y.to(device)
    z = backbone(x)
    h_stack = torch.stack([expert(z) for expert in moe.experts], dim=1)
    pi = F.softmax(moe.router(z), dim=-1)
    zbar = (pi.unsqueeze(-1) * h_stack).sum(dim=1)

    expert_logits, expert_probs = expert_logits_and_probs(moe, h_stack)
    routed_logits = moe.classifier(zbar)
    routed_probs = F.softmax(routed_logits, dim=-1)

    p = expert_probs.clamp_min(EPS)
    p_mean = p.mean(dim=1).clamp_min(EPS)
    kl = (p * (p.log() - p_mean.unsqueeze(1).log())).sum(dim=-1)
    s_pred = kl.mean(dim=1)

    num_classes = expert_logits.size(-1)
    if num_classes == 2:
        signed_y = (2 * y - 1).float()
        margins = signed_y.unsqueeze(1) * (
            expert_logits[:, :, 1] - expert_logits[:, :, 0]
        )
    else:
        true_logits = expert_logits.gather(
            2, y.view(-1, 1, 1).expand(-1, expert_logits.size(1), 1)
        ).squeeze(-1)
        masked_logits = expert_logits.clone()
        masked_logits.scatter_(
            2,
            y.view(-1, 1, 1).expand(-1, expert_logits.size(1), 1),
            -torch.inf,
        )
        other_logits = masked_logits.max(dim=2).values
        margins = true_logits - other_logits
    s_margin = torch.var(margins, dim=1, unbiased=False)

    pbar_true = routed_probs.gather(1, y.view(-1, 1)).squeeze(1).clamp_min(EPS)
    loss = -pbar_true.log()
    router_entropy = -(pi.clamp_min(EPS) * pi.clamp_min(EPS).log()).sum(dim=1)

    return {
        "zbar": zbar.detach().cpu(),
        "s_pred": s_pred.detach().cpu(),
        "s_margin": s_margin.detach().cpu(),
        "loss": loss.detach().cpu(),
        "router_entropy": router_entropy.detach().cpu(),
    }


@torch.no_grad()
def score_train_set(backbone, moe, loader, device):
    rows = {
        "labels": [],
        "groups": [],
        "places": [],
        "metadata_indices": [],
        "zbar": [],
        "s_pred": [],
        "s_margin": [],
        "loss": [],
        "router_entropy": [],
    }
    backbone.eval()
    moe.eval()
    for step, (x, y, group, place, metadata_idx) in enumerate(loader):
        scored = score_batch(backbone, moe, x, y, device)
        rows["labels"].append(y.cpu())
        rows["groups"].append(group.cpu())
        rows["places"].append(place.cpu())
        rows["metadata_indices"].append(metadata_idx.cpu())
        for key in ["zbar", "s_pred", "s_margin", "loss", "router_entropy"]:
            rows[key].append(scored[key])
        if step % 20 == 0:
            print(f"[score] batch {step}")

    return {
        key: torch.cat(value, dim=0)
        for key, value in rows.items()
    }


def zscore(values):
    mean = values.mean()
    std = values.std(unbiased=False).clamp_min(EPS)
    return (values - mean) / std


def add_scores(rows, hparams):
    z_pred = zscore(rows["s_pred"])
    z_margin = zscore(rows["s_margin"])
    z_loss = zscore(rows["loss"])
    z_router = zscore(rows["router_entropy"])
    s_ed = (
        hparams.alpha_pred * z_pred
        + hparams.beta_margin * z_margin
        + hparams.gamma_loss * z_loss
        + hparams.delta_router * z_router
    )
    rows["z_s_pred"] = z_pred
    rows["z_s_margin"] = z_margin
    rows["z_loss"] = z_loss
    rows["z_router_entropy"] = z_router
    rows["s_ed"] = s_ed


def ranks_desc(values):
    order = torch.argsort(values, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, values.numel() + 1, dtype=order.dtype)
    return ranks


def add_ranks(rows):
    labels = rows["labels"]
    s_ed = rows["s_ed"]
    rows["rank_s_ed_global"] = ranks_desc(s_ed)
    within = torch.zeros_like(rows["rank_s_ed_global"])
    for y in torch.unique(labels).tolist():
        mask = labels == int(y)
        local_ranks = ranks_desc(s_ed[mask])
        within[mask] = local_ranks
    rows["rank_s_ed_within_class"] = within


def normalize_features(features):
    return F.normalize(features.float(), p=2, dim=1, eps=EPS)


def pairwise_similarity(features_a, features_b, similarity_mode):
    sim = features_a @ features_b.T
    if similarity_mode == "cosine_clamped":
        return sim.clamp_min(0.0)
    if similarity_mode == "cosine":
        return sim
    raise ValueError(f"unknown similarity_mode: {similarity_mode}")


def select_probe(labels, features, scores, hparams):
    if hparams.use_group_labels_for_selection:
        raise ValueError("group labels are not allowed for probe selection")
    if hparams.selection_method != "ed_sps":
        raise ValueError(f"unknown selection_method: {hparams.selection_method}")
    if hparams.class_balance != "equal":
        raise ValueError(f"unknown class_balance: {hparams.class_balance}")
    if hparams.coverage_universe != "candidate_pool":
        raise ValueError("only coverage_universe='candidate_pool' is implemented")

    labels = labels.long()
    features = normalize_features(features)
    num_classes = int(labels.max().item()) + 1
    n_train = int(labels.numel())
    requested_budget = math.floor(hparams.probe_percentage * n_train)
    per_class_budget = requested_budget // num_classes
    actual_budget = num_classes * per_class_budget
    if per_class_budget < 1:
        raise ValueError("Probe budget too small: per-class budget is 0.")

    selected_positions = []
    selected_records = []

    for class_id in range(num_classes):
        class_positions = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if class_positions.numel() < per_class_budget:
            raise ValueError(
                f"class {class_id} has {class_positions.numel()} samples, fewer than "
                f"per_class_budget={per_class_budget}"
            )

        ranked = class_positions[
            torch.argsort(scores[class_positions], descending=True, stable=True)
        ]
        pool_size = min(
            int(class_positions.numel()),
            int(hparams.candidate_pool_multiplier * per_class_budget),
        )
        candidates = ranked[:pool_size]
        candidate_features = features[candidates]
        sim_pool = pairwise_similarity(
            candidate_features, candidate_features, hparams.similarity_mode
        )

        candidate_scores = scores[candidates]
        available = torch.ones(pool_size, dtype=torch.bool)
        coverage_current = torch.zeros(pool_size)
        selected_local = []

        for selection_rank in range(1, per_class_budget + 1):
            best_local = None
            best_score = None
            best_gain = None
            best_redundancy = None

            for local_idx in torch.nonzero(available, as_tuple=False).flatten().tolist():
                sim_to_i = sim_pool[:, local_idx]
                if hparams.use_coverage:
                    gain = (
                        torch.maximum(coverage_current, sim_to_i) - coverage_current
                    ).mean()
                else:
                    gain = torch.tensor(0.0)

                if hparams.use_redundancy and selected_local:
                    redundancy = sim_pool[local_idx, selected_local].max()
                else:
                    redundancy = torch.tensor(0.0)

                greedy_score = (
                    candidate_scores[local_idx]
                    + hparams.coverage_weight * gain
                    - hparams.redundancy_weight * redundancy
                )
                if best_score is None or greedy_score.item() > best_score:
                    best_local = local_idx
                    best_score = greedy_score.item()
                    best_gain = gain.item()
                    best_redundancy = redundancy.item()

            if best_local is None:
                raise RuntimeError(f"no candidate available for class {class_id}")

            available[best_local] = False
            selected_local.append(best_local)
            coverage_current = torch.maximum(coverage_current, sim_pool[:, best_local])
            train_position = int(candidates[best_local].item())
            selected_positions.append(train_position)
            selected_records.append({
                "train_position": train_position,
                "label": int(class_id),
                "selection_rank_within_class": int(selection_rank),
                "greedy_score_at_selection": float(best_score),
                "coverage_gain_at_selection": float(best_gain),
                "redundancy_at_selection": float(best_redundancy),
                "s_ED": float(scores[train_position].item()),
            })

    return {
        "selected_positions": selected_positions,
        "selected_records": selected_records,
        "requested_budget": requested_budget,
        "per_class_budget": per_class_budget,
        "actual_budget": actual_budget,
    }


def bincount_dict(values, minlength):
    counts = np.bincount(values.astype(np.int64), minlength=minlength)
    return {str(i): int(v) for i, v in enumerate(counts.tolist())}


def compute_group_stats(selected_positions, labels, groups, places, num_classes, n_groups):
    selected = np.asarray(selected_positions, dtype=np.int64)
    labels_np = labels.numpy().astype(np.int64)
    groups_np = groups.numpy().astype(np.int64)
    places_np = places.numpy().astype(np.int64)

    if not set(np.unique(labels_np).tolist()).issubset({0, 1}):
        raise ValueError("Waterbirds conflict reporting expects labels in {0, 1}")
    if not set(np.unique(places_np).tolist()).issubset({0, 1}):
        raise ValueError("Waterbirds conflict reporting expects places in {0, 1}")

    selected_labels = labels_np[selected]
    selected_groups = groups_np[selected]
    selected_places = places_np[selected]

    class_counts = bincount_dict(selected_labels, num_classes)
    group_counts = bincount_dict(selected_groups, n_groups)

    conflict_fraction = {}
    gaps = []
    for class_id in range(num_classes):
        mask = selected_labels == class_id
        if mask.sum() == 0:
            q_y = None
        else:
            conflict = selected_places[mask] != selected_labels[mask]
            q_y = float(conflict.mean())
            gaps.append(abs(q_y - 0.5))
        conflict_fraction[str(class_id)] = q_y

    mean_abs_oracle_imbalance = float(np.mean(gaps)) if gaps else None
    return {
        "selected_class_counts": class_counts,
        "selected_group_counts": group_counts,
        "conflict_fraction_per_class": conflict_fraction,
        "mean_abs_oracle_imbalance": mean_abs_oracle_imbalance,
        "waterbirds_label_mapping": {
            "y": "0=landbird, 1=waterbird",
            "place": "0=land, 1=water",
            "conflict_rule": "place != y",
        },
    }


def tensor_float(value):
    return float(value.item())


def tensor_int(value):
    return int(value.item())


def build_score_records(rows, selected_positions):
    selected_set = set(int(p) for p in selected_positions)
    records = []
    n = int(rows["labels"].numel())
    for train_position in range(n):
        records.append({
            "metadata_index": tensor_int(rows["metadata_indices"][train_position]),
            "train_position": int(train_position),
            "label": tensor_int(rows["labels"][train_position]),
            "group": tensor_int(rows["groups"][train_position]),
            "place": tensor_int(rows["places"][train_position]),
            "s_pred": tensor_float(rows["s_pred"][train_position]),
            "s_margin": tensor_float(rows["s_margin"][train_position]),
            "loss": tensor_float(rows["loss"][train_position]),
            "router_entropy": tensor_float(rows["router_entropy"][train_position]),
            "z_s_pred": tensor_float(rows["z_s_pred"][train_position]),
            "z_s_margin": tensor_float(rows["z_s_margin"][train_position]),
            "z_loss": tensor_float(rows["z_loss"][train_position]),
            "z_router_entropy": tensor_float(rows["z_router_entropy"][train_position]),
            "s_ED": tensor_float(rows["s_ed"][train_position]),
            "rank_s_ed_global": tensor_int(rows["rank_s_ed_global"][train_position]),
            "rank_s_ed_within_class": tensor_int(
                rows["rank_s_ed_within_class"][train_position]
            ),
            "selected": train_position in selected_set,
        })
    return records


def finite_check(rows):
    for key in [
        "s_pred",
        "s_margin",
        "loss",
        "router_entropy",
        "z_s_pred",
        "z_s_margin",
        "z_loss",
        "z_router_entropy",
        "s_ed",
    ]:
        if not torch.isfinite(rows[key]).all():
            raise ValueError(f"non-finite values found in {key}")


def write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/deit_small_moe_waterbirds.pth")
    ap.add_argument(
        "--data-root",
        default="./data/waterbirds/waterbird_complete95_forest2water2",
        help="path to waterbird_complete95_forest2water2 with metadata.csv",
    )
    ap.add_argument(
        "--output",
        default="probes/deit_small_moe_waterbirds_ed_sps_probe.json",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--probe-percentage", type=float, default=0.05)
    ap.add_argument("--selection-method", default="ed_sps")
    ap.add_argument("--score-normalization", default="zscore")
    ap.add_argument("--alpha-pred", type=float, default=1.0)
    ap.add_argument("--beta-margin", type=float, default=0.0)
    ap.add_argument("--gamma-loss", type=float, default=1.0)
    ap.add_argument("--delta-router", type=float, default=1.0)
    ap.add_argument("--class-balance", default="equal")
    ap.add_argument("--use-coverage", action="store_true", default=True)
    ap.add_argument("--no-coverage", dest="use_coverage", action="store_false")
    ap.add_argument("--coverage-weight", type=float, default=0.2)
    ap.add_argument("--coverage-similarity", default="cosine")
    ap.add_argument("--coverage-universe", default="candidate_pool")
    ap.add_argument("--similarity-mode", default="cosine_clamped")
    ap.add_argument("--use-redundancy", action="store_true", default=True)
    ap.add_argument("--no-redundancy", dest="use_redundancy", action="store_false")
    ap.add_argument("--redundancy-weight", type=float, default=0.1)
    ap.add_argument("--candidate-pool-multiplier", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hparams = SelectionHparams(
        probe_percentage=args.probe_percentage,
        selection_method=args.selection_method,
        score_normalization=args.score_normalization,
        alpha_pred=args.alpha_pred,
        beta_margin=args.beta_margin,
        gamma_loss=args.gamma_loss,
        delta_router=args.delta_router,
        class_balance=args.class_balance,
        use_coverage=args.use_coverage,
        coverage_weight=args.coverage_weight,
        coverage_similarity=args.coverage_similarity,
        coverage_universe=args.coverage_universe,
        similarity_mode=args.similarity_mode,
        use_redundancy=args.use_redundancy,
        redundancy_weight=args.redundancy_weight,
        candidate_pool_multiplier=args.candidate_pool_multiplier,
        use_group_labels_for_selection=False,
        use_group_labels_for_eval=True,
        seed=args.seed,
    )
    if hparams.score_normalization != "zscore":
        raise ValueError("only score_normalization='zscore' is implemented")
    if hparams.coverage_similarity != "cosine":
        raise ValueError("only coverage_similarity='cosine' is implemented")

    ckpt = load_stage1(args.checkpoint, device)
    cfg = ckpt["config"]
    print(
        f"[checkpoint] backbone={cfg['backbone']} embed_dim={cfg['embed_dim']} "
        f"num_experts={cfg['num_experts']} num_classes={cfg['num_classes']}"
    )

    base = MyWaterBirdsDataset(
        args.data_root, remove_minority_groups=False, transform=build_transforms()
    )
    train_ds = Subset(base, base.train_idxs)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    print(f"[dataset] train={len(train_ds)} num_classes={base.n_classes} n_groups={base.n_groups}")

    backbone = build_backbone(cfg, ckpt, device)
    moe = build_moe(cfg, ckpt, device)
    validate_model_config(cfg, backbone, moe, base)

    rows = score_train_set(backbone, moe, loader, device)
    add_scores(rows, hparams)
    add_ranks(rows)
    finite_check(rows)

    selection = select_probe(
        labels=rows["labels"],
        features=rows["zbar"],
        scores=rows["s_ed"],
        hparams=hparams,
    )
    selected_positions = selection["selected_positions"]
    selected_indices = [
        tensor_int(rows["metadata_indices"][position])
        for position in selected_positions
    ]
    for record in selection["selected_records"]:
        position = record["train_position"]
        record["metadata_index"] = tensor_int(rows["metadata_indices"][position])

    if len(selected_indices) != selection["actual_budget"]:
        raise ValueError(
            f"selected {len(selected_indices)} samples but actual_budget is "
            f"{selection['actual_budget']}"
        )
    if not set(selected_indices).issubset(set(int(i) for i in base.train_idxs.tolist())):
        raise ValueError("selected index outside base.train_idxs")

    group_stats = compute_group_stats(
        selected_positions=selected_positions,
        labels=rows["labels"],
        groups=rows["groups"],
        places=rows["places"],
        num_classes=base.n_classes,
        n_groups=base.n_groups,
    )
    expected_count = selection["per_class_budget"]
    for class_id, count in group_stats["selected_class_counts"].items():
        if count != expected_count:
            raise ValueError(
                f"class {class_id} selected count {count}, expected {expected_count}"
            )

    score_records = build_score_records(rows, selected_positions)
    if len(score_records) != len(train_ds):
        raise ValueError("score record count does not match train set size")

    payload = {
        "selected_indices": selected_indices,
        "selected_records": selection["selected_records"],
        "requested_budget": selection["requested_budget"],
        "actual_budget": selection["actual_budget"],
        "per_class_budget": selection["per_class_budget"],
        "scores": score_records,
        "selected_class_counts": group_stats["selected_class_counts"],
        "selected_group_counts": group_stats["selected_group_counts"],
        "conflict_fraction_per_class": group_stats["conflict_fraction_per_class"],
        "mean_abs_oracle_imbalance": group_stats["mean_abs_oracle_imbalance"],
        "waterbirds_label_mapping": group_stats["waterbirds_label_mapping"],
        "hparams": hparams.__dict__,
        "checkpoint": {
            "path": args.checkpoint,
            "config": cfg,
            "epoch": ckpt.get("epoch"),
            "val_avg": ckpt.get("val_avg"),
            "val_worst": ckpt.get("val_worst"),
        },
        "dataset": {
            "data_root": args.data_root,
            "n_train": len(train_ds),
            "num_classes": base.n_classes,
            "n_groups": base.n_groups,
            "train_indices_are_metadata_indices": True,
        },
    }
    write_json(args.output, payload)
    print(
        f"[done] wrote {args.output} with actual_budget={selection['actual_budget']} "
        f"requested_budget={selection['requested_budget']}"
    )


if __name__ == "__main__":
    main()
