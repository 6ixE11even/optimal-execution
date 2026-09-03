"""
A policy-gradient execution agent, written out rather than imported.

Two hidden tanh layers, a Gaussian head, and a squashing function that maps the
sample into [0, 1] - the fraction of remaining inventory to sell in this bar. Trained
with REINFORCE and a learned value baseline. The gradients are derived here instead
of coming from an autodiff library, which is a deliberate choice: the whole repo runs
on numpy, and the backward pass for a three-layer network with a Gaussian head is
twenty lines that a reader can check against the forward pass.

The objective is the one Almgren-Chriss minimises, E[shortfall] + lambda*Var, so the
closed form and the agent are optimising the same thing and the comparison in
`scripts/train_rl_execution.py` is a fair one. An agent rewarded on cost alone would
"beat" Almgren-Chriss by holding inventory longer and carrying the risk the objective
was supposed to price.
"""
from __future__ import annotations

import numpy as np

from execution.env import (EpisodeSpec, N_FEATURES, features, run_schedule,
                           twap_fractions)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class GaussianPolicy:
    """pi(a|s): tanh MLP -> mean of a Gaussian in pre-squash space -> sigmoid."""

    def __init__(self, n_in: int = N_FEATURES, hidden: int = 32, seed: int = 0,
                 log_std: float = -1.0):
        rng = np.random.default_rng(seed)
        scale = lambda a, b: rng.normal(0, np.sqrt(2.0 / a), (a, b))   # noqa: E731
        self.W1, self.b1 = scale(n_in, hidden), np.zeros(hidden)
        self.W2, self.b2 = scale(hidden, hidden), np.zeros(hidden)
        self.W3, self.b3 = scale(hidden, 1) * 0.01, np.zeros(1)
        self.log_std = np.array([log_std])

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.log_std]

    def forward(self, s):
        h1 = np.tanh(s @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        mu = (h2 @ self.W3 + self.b3)[0]
        return mu, (h1, h2)

    def act(self, s, rng, greedy: bool = False):
        mu, cache = self.forward(s)
        std = float(np.exp(self.log_std[0]))
        z = mu if greedy else mu + std * rng.standard_normal()
        return _sigmoid(z), z, mu, cache

    def grad_log_prob(self, s, z, mu, cache):
        """d log pi(z|s) / d theta for a Gaussian in the pre-squash variable."""
        h1, h2 = cache
        std = float(np.exp(self.log_std[0]))
        d_mu = (z - mu) / std ** 2
        g = {}
        g["W3"] = np.outer(h2, [d_mu])
        g["b3"] = np.array([d_mu])
        d_h2 = self.W3[:, 0] * d_mu * (1 - h2 ** 2)
        g["W2"] = np.outer(h1, d_h2)
        g["b2"] = d_h2
        d_h1 = (self.W2 @ d_h2) * (1 - h1 ** 2)
        g["W1"] = np.outer(s, d_h1)
        g["b1"] = d_h1
        g["log_std"] = np.array([((z - mu) ** 2 / std ** 2) - 1.0])
        return g


class ValueBaseline:
    """A linear value function on the same features, fit online by least squares.

    A baseline does not bias the policy gradient - E[grad log pi] is zero - but it
    takes the variance of the estimate down by roughly the amount of the return that
    is explained by the state rather than by the action. Execution returns are
    dominated by where the price happened to go, so the baseline is doing most of the
    work of making this trainable at all.
    """

    def __init__(self, n_in: int = N_FEATURES, lr: float = 0.01):
        self.w = np.zeros(n_in + 1)
        self.lr = lr

    def _x(self, s):
        return np.concatenate([s, [1.0]])

    def predict(self, s):
        return float(self._x(s) @ self.w)

    def update(self, s, target):
        x = self._x(s)
        self.w += self.lr * (target - x @ self.w) * x


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p += self.lr * mh / (np.sqrt(vh) + self.eps)


def control_variate(spec: EpisodeSpec) -> np.ndarray:
    """Per-step cost of the TWAP schedule on this same price path, in basis points.

    Almost all of an execution's realised cost is where the price happened to go, and
    that part is common to every schedule facing the same path. Subtracting a fixed
    schedule's cost on the identical path removes it. The subtraction is legitimate
    because it does not depend on the agent's action, so it shifts the reward without
    touching the gradient's expectation - the textbook definition of a control variate.

    The size of the problem: the difference between the closed form and TWAP is a
    tenth of a basis point, and the standard deviation of a single episode's outcome
    is eighty. Without this, REINFORCE is being asked to hear one part in eight
    hundred through the noise.
    """
    out = run_schedule(spec, twap_fractions(spec.N))
    risk = spec.lam * spec.sigma ** 2 * spec.tau * out["holdings"] ** 2
    return (out["cost_steps"] + risk) * (1e4 / out["notional"])


def rollout(policy: GaussianPolicy, spec: EpisodeSpec, rng, greedy: bool = False,
            control: np.ndarray | None = None):
    """One episode. Returns the trajectory and the per-step rewards in basis points."""
    states, zs, mus, caches, fracs = [], [], [], [], []
    x = spec.X
    for k in range(spec.N):
        s = features(spec, k, x)
        a, z, mu, cache = policy.act(s, rng, greedy=greedy)
        if k == spec.N - 1:
            a = 1.0
        states.append(s); zs.append(z); mus.append(mu); caches.append(cache); fracs.append(a)
        x -= x * a
    out = run_schedule(spec, np.asarray(fracs))
    bps = 1e4 / out["notional"]
    step_risk = spec.lam * spec.sigma ** 2 * spec.tau * out["holdings"] ** 2
    rewards = -(out["cost_steps"] + step_risk) * bps
    if control is not None:
        rewards = rewards + control
    return {"states": states, "zs": zs, "mus": mus, "caches": caches,
            "fractions": np.asarray(fracs), "rewards": rewards, **out}


def train(policy: GaussianPolicy, episodes, epochs: int = 6, lr: float = 3e-3,
          batch: int = 32, seed: int = 0, log_every: int = 0, control: bool = True):
    """REINFORCE with a value baseline over the training episodes."""
    rng = np.random.default_rng(seed)
    cv = [control_variate(e) if control else None for e in episodes]
    baseline = ValueBaseline()
    opt = Adam(policy.params(), lr=lr)
    keys = ["W1", "b1", "W2", "b2", "W3", "b3", "log_std"]
    history = []

    order = np.arange(len(episodes))
    for epoch in range(epochs):
        rng.shuffle(order)
        acc = [np.zeros_like(p) for p in policy.params()]
        n_in_batch, epoch_obj = 0, []
        for j, idx in enumerate(order, 1):
            spec = episodes[idx]
            tr = rollout(policy, spec, rng, control=cv[idx])
            returns = np.cumsum(tr["rewards"][::-1])[::-1]        # undiscounted
            for k in range(spec.N - 1):                            # last step is forced
                s = tr["states"][k]
                adv = returns[k] - baseline.predict(s)
                baseline.update(s, returns[k])
                g = policy.grad_log_prob(s, tr["zs"][k], tr["mus"][k], tr["caches"][k])
                for i, key in enumerate(keys):
                    acc[i] += adv * g[key]
            epoch_obj.append(tr["objective"] / tr["notional"] * 1e4)
            n_in_batch += 1
            if n_in_batch == batch:
                opt.step([a / batch for a in acc])
                acc = [np.zeros_like(p) for p in policy.params()]
                n_in_batch = 0
            if log_every and j % log_every == 0:
                print(f"    epoch {epoch + 1} ep {j}/{len(order)}  "
                      f"objective {np.mean(epoch_obj[-log_every:]):.2f} bps")
        history.append(float(np.mean(epoch_obj)))
    return history
