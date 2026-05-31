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
        l_div = loss_div(h_stack)
        l_sp = loss_sparse(pi)
        l_bal = loss_balance(pi)
        loss = (
            l_cls
            + self.lambda_div * l_div
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
            "loss_div": l_div.item(),
            "loss_sp": l_sp.item(),
            "loss_bal": l_bal.item(),
            "accuracy": accuracy.item(),
            "preds": logits,
        }
