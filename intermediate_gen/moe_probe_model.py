''' MoE probe head: experts + router + classifier. Drop-in replacement for LinearProbeModel. '''

import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_div(h_stack: torch.Tensor) -> torch.Tensor:
    '''
        Force per-expert outputs orthogonal.
        h_stack: (B, M, r) tensor of expert outputs.
        Returns sum over m != n of ||(H_m^T H_n) / B||_F^2.
    '''
    B, M, _ = h_stack.shape
    total = h_stack.new_zeros(())
    for m in range(M):
        for n in range(M):
            if m == n:
                continue
            C = (h_stack[:, m, :].T @ h_stack[:, n, :]) / B
            total = total + (C ** 2).sum()
    return total


def loss_sparse(pi: torch.Tensor) -> torch.Tensor:
    '''
        Mean Shannon entropy of per-token routing distribution.
        Minimizing this loss pushes pi toward one-hot (sparse routing).
        pi: (B, M) softmax-normalized.
    '''
    entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)
    return entropy.mean()


def loss_balance(pi: torch.Tensor) -> torch.Tensor:
    '''
        Penalize deviation of mean routing probabilities from uniform 1/M.
        pi: (B, M) softmax-normalized.
    '''
    M = pi.size(1)
    mean = pi.mean(dim=0)
    return ((mean - 1.0 / M) ** 2).sum()


def loss_expert_ce(expert_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    '''
        Mean cross-entropy over each expert's standalone classifier logits.
        expert_logits: (B, M, C)
        y: (B,)
    '''
    B, M, C = expert_logits.shape
    targets = y[:, None].expand(B, M).reshape(B * M)
    return F.cross_entropy(expert_logits.reshape(B * M, C), targets)


def _residual_offdiag_terms(
    expert_logits: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
):
    '''
        Return class-local squared off-diagonal correlations and mean abs
        off-diagonal correlations for residual logit patterns.
    '''
    B, M, C = expert_logits.shape
    zero = expert_logits.new_zeros(())
    if B < 2 or M < 2 or C < 2:
        return [zero], [zero]

    true_logits = expert_logits.gather(
        dim=2,
        index=y[:, None, None].expand(-1, M, 1),
    )
    residual = expert_logits - true_logits
    terms = []
    diag_terms = []

    for label in y.unique(sorted=True):
        mask = y == label
        if mask.sum().item() < 2:
            continue
        class_residual = residual[mask]  # (B_y, M, C)
        non_true = torch.ones(C, dtype=torch.bool, device=expert_logits.device)
        non_true[int(label.item())] = False
        class_residual = class_residual[:, :, non_true]
        if class_residual.numel() == 0:
            continue

        R = class_residual.permute(1, 0, 2).reshape(M, -1)
        if R.size(1) < 2:
            continue
        R = R - R.mean(dim=1, keepdim=True)
        std = R.std(dim=1, keepdim=True, unbiased=False)
        if (std <= eps).all():
            continue
        R = R / (std + eps)
        corr = (R @ R.T) / R.size(1)
        off_diag = corr - torch.diag(torch.diag(corr))
        denom = max(M * (M - 1), 1)
        terms.append((off_diag ** 2).sum() / denom)
        diag_terms.append(off_diag.abs().sum() / denom)

    if not terms:
        return [zero], [zero]
    return terms, diag_terms


def loss_residual_logit_diversity(
    expert_logits: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    '''
        Penalize correlation between experts' class-local residual-logit patterns.
        Residuals are logit_c - logit_y with the true-class column excluded.
    '''
    terms, _ = _residual_offdiag_terms(expert_logits, y, eps=eps)
    return torch.stack(terms).mean()


@torch.no_grad()
def residual_corr_offdiag_mean_abs(
    expert_logits: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    '''
        Diagnostic: mean absolute off-diagonal residual-logit correlation.
    '''
    _, terms = _residual_offdiag_terms(expert_logits, y, eps=eps)
    return torch.stack(terms).mean()


class LoadBalanceLoss(nn.Module):
    '''
        Backward-compatible wrapper for batch load balancing.
        L_bal = sum_m (E_batch[pi_m] - 1/M)^2.
    '''

    def __init__(self, num_experts: int, beta: float = 0.99):
        super().__init__()
        self.num_experts = num_experts
        self.beta = beta

    def forward(self, pi: torch.Tensor) -> torch.Tensor:
        return loss_balance(pi)


class MoEProbeModel(nn.Module):
    '''
        Drop-in replacement for LinearProbeModel.
        forward(x) -> (logits, pi, h_stack)
        step_loss(x, y) trains the probe with cross-entropy + lambda_div * loss_div
        + lambda_sp * loss_sparse + lambda_bal * loss_balance.
    '''

    def __init__(
        self,
        input_size,
        output_size,
        num_experts=4,
        lr=1e-4,
        reg_type="l2",
        reg_weight=0.0,
        lambda_div=0.02,
        lambda_sp=0.02,
        lambda_bal=0.02,
        lambda_expert_ce=0.0,
        lambda_res_div=0.0,
        bal_beta=0.99,
        optimizer="adam",
        sgd_momentum=0.0,
    ):
        super().__init__()
        hidden = 4 * input_size
        self.num_experts = num_experts
        self.lambda_div = lambda_div
        self.lambda_sp = lambda_sp
        self.lambda_bal = lambda_bal
        self.lambda_expert_ce = lambda_expert_ce
        self.lambda_res_div = lambda_res_div
        self.reg_type = reg_type
        self.reg_weight = reg_weight
        self.lr = lr

        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_size, hidden),
                nn.GELU(),
                nn.Linear(hidden, input_size),
            )
            for _ in range(num_experts)
        )
        self.router = nn.Linear(input_size, num_experts, bias=True)
        self.classifier = nn.Linear(input_size, output_size)
        self.balance_loss_fn = LoadBalanceLoss(num_experts, beta=bal_beta)

        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.parameters(), lr=lr, momentum=sgd_momentum
            )
        elif optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"unknown optimizer: {optimizer!r}")

    def forward(self, x):
        h_stack = torch.stack([E(x) for E in self.experts], dim=1)
        pi = F.softmax(self.router(x), dim=-1)
        h = (pi.unsqueeze(-1) * h_stack).sum(dim=1)
        logits = self.classifier(h)
        return logits, pi, h_stack

    def step_loss(self, x, y):
        '''
            x: representation from the backbone at the layer of interest (batch_size, feature_size)
            y: labels (batch_size)
        '''
        logits, pi, h_stack = self.forward(x)
        accuracy = (logits.argmax(dim=1) == y).float().mean()
        l_cls = F.cross_entropy(logits, y)
        B, M, D = h_stack.shape
        expert_logits = self.classifier(h_stack.reshape(B * M, D)).reshape(B, M, -1)
        l_expert_ce = loss_expert_ce(expert_logits, y)
        l_div = loss_div(h_stack)
        l_res_div = loss_residual_logit_diversity(expert_logits, y)
        l_sp = loss_sparse(pi)
        l_bal = loss_balance(pi)
        loss = (
            l_cls
            + self.lambda_expert_ce * l_expert_ce
            + self.lambda_div * l_div
            + self.lambda_res_div * l_res_div
            + self.lambda_sp * l_sp
            + self.lambda_bal * l_bal
        )
        if self.reg_type == "l2":
            loss = loss + self.reg_weight * torch.norm(self.classifier.weight, p=2)
        elif self.reg_type == "l1":
            loss = loss + self.reg_weight * torch.norm(self.classifier.weight, p=1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "loss_cls": l_cls.item(),
            "loss_routed_ce": l_cls.item(),
            "loss_expert_ce": l_expert_ce.item(),
            "loss_div": l_div.item(),
            "loss_res_div": l_res_div.item(),
            "loss_sp": l_sp.item(),
            "loss_bal": l_bal.item(),
            "accuracy": accuracy.item(),
            "preds": logits,
        }
