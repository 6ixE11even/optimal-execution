# Optimal Execution — Almgren–Chriss

Liquidate a large position the smart way. Trade too fast and you pay market impact;
too slow and price risk eats you alive. **Almgren–Chriss** makes that trade-off exact —
minimise `E[cost] + λ·Var[cost]` — and the optimal schedule comes out in closed form.

![Optimal liquidation trajectory](reports/figures/trajectory.png)

## The trade-off

On the example book (liquidate 1,000,000 units over 5 days in 20 slices):

| Schedule | Expected cost | Risk (std of cost) |
|---|--:|--:|
| **Almgren–Chriss** | 2,355,346 | 832,878 |
| TWAP (equal slices) | 681,250 | 2,484,955 |

TWAP is cheaper on impact but carries **3× the risk** — it holds inventory longer.
Almgren–Chriss front-loads the selling to cut that risk, and as risk-aversion `λ` rises
it liquidates faster still (shorter half-life). Sweeping `λ` traces the **efficient
frontier**: the best achievable cost for each level of risk.

![Efficient frontier](reports/figures/efficient_frontier.png)

## Method (`model.py`)

The optimal holdings decay as `x(t) = X · sinh(κ(T−t)) / sinh(κT)`, where the rate
`κ` grows with `λ·σ²/η` — more risk or more aversion ⇒ faster exit. Expected cost sums
permanent impact, the half-spread, and temporary impact `η·(rate)²`; variance is
`σ²·Σ τ·x²`. `frontier.py` sweeps `λ`; `viz.py` plots both charts.

## The math

Sell $X$ shares over $[0,T]$ in $N$ slices of length $\tau$, holdings $x_k$, trade
rate $v_k = (x_{k-1}-x_k)/\tau$. Price follows an arithmetic random walk with
permanent impact $\gamma$, and each trade executes at a temporary-impact-adjusted
price $\tilde S_k = S_{k-1} - \eta v_k$ (plus half-spread $\epsilon$). Implementation
shortfall then has

$$E[C] = \tfrac{1}{2}\gamma X^2 + \epsilon X + \eta \tau \sum_k v_k^2,
\qquad
\text{Var}[C] = \sigma^2 \tau \sum_k x_k^2$$

Minimising the mean-variance objective $E[C] + \lambda \, \text{Var}[C]$ over
trajectories gives a discrete sinh solution,

$$x_j = X \, \frac{\sinh\!\big(\kappa (T - t_j)\big)}{\sinh(\kappa T)},
\qquad
\kappa^2 \approx \frac{\lambda \sigma^2}{\eta}$$

so the position decays exponentially with urgency $\kappa$: more variance penalty
($\lambda$), more volatility, or cheaper impact all mean sell faster. $\lambda \to 0$
recovers TWAP ($v$ constant); sweeping $\lambda$ traces the efficient frontier, which
is convex — its slope at any point is $-1/\lambda$.

## References

- Almgren, R. & Chriss, N. (2001), *Optimal Execution of Portfolio Transactions*, Journal of Risk 3(2).
- Bertsimas, D. & Lo, A. (1998), *Optimal Control of Execution Costs*, Journal of Financial Markets 1(1).
- Gatheral, J. (2010), *No-Dynamic-Arbitrage and Market Impact*, Quantitative Finance 10(7) — why impact functions can't be arbitrary.

## Real-data calibration (`data.py`)

`σ` and average daily volume are pulled from a **real Deribit instrument** so the inputs
aren't invented; temporary impact scales as `1/ADV` (deeper books cost less to trade):

```bash
uv run python scripts/run_execution.py              # example parameters (offline)
uv run python scripts/run_execution.py --calibrate  # real sigma/ADV from Deribit (BTC-PERPETUAL)
uv run pytest                                       # trajectory + frontier invariants
```

## Structure

```
optimal-execution/
├── src/execution/
│   ├── model.py      # Almgren-Chriss closed form (trajectory, cost, variance, TWAP)
│   ├── frontier.py   # efficient frontier over risk-aversion
│   ├── data.py       # calibrate sigma/ADV from real Deribit prices
│   └── viz.py        # trajectory + frontier charts
├── scripts/run_execution.py
└── tests/            # liquidates fully, beats TWAP on E+lambda*Var, faster when risk-averse
```

---

*Built by Tejas Pandya — NYU MSFE.*
