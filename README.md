# Optimal Execution — Almgren–Chriss

Liquidate a large position the smart way. Trade too fast and you pay market impact;
too slow and price risk eats you alive. **Almgren–Chriss** makes that trade-off exact —
minimise `E[cost] + λ·Var[cost]` — and the optimal schedule comes out in closed form.

![Optimal liquidation trajectory](reports/figures/trajectory.png)

## The trade-off

Everything below is calibrated to a live Deribit book, not to chosen numbers.
BTC-PERPETUAL on 2026-09-02: price $77,019, σ $1,494/day, ADV 5,377 contracts,
touch spread $4.00. Liquidating over 5 days in 20 slices at a 1-day half-life:

| order size | schedule | cost ($) | cost (bps) | risk, std ($) |
|---|---|--:|--:|--:|
| 2% of ADV | Almgren–Chriss | 1,410 | 1.7 | 124,003 |
| ($8.3M) | TWAP | 930 | 1.1 | 199,707 |
| 10% of ADV | **Almgren–Chriss** | 32,016 | **7.7** | 620,009 |
| ($41.4M) | TWAP | 20,025 | 4.8 | 998,522 |
| 25% of ADV | Almgren–Chriss | 199,592 | 19.3 | 1,549,986 |
| ($103.5M) | TWAP | 124,648 | 12.0 | 2,496,246 |

TWAP is cheaper on impact but carries **~60% more risk** — it holds inventory longer.
Almgren–Chriss front-loads the selling to cut that risk, paying about 60% more in
impact to take roughly 38% off the standard deviation. Which of those you want is the
whole question, and sweeping urgency traces the **efficient frontier**: the best
achievable cost at each level of risk.

Two things the calibration forces you to get right, and neither is optional:

**Size the order off the book you measured.** Liquidating a hardcoded 1,000,000 units
against a 5,377-contract ADV is 186 days of volume, and a linear temporary-impact
model describes nothing at that size. Orders are specified as a participation rate.

**Impact has units.** `η = 1/ADV` carries units of 1/(contracts per day), and on a
$77,000 instrument it produced two cents of slippage per contract — 0.0 bps on a
$41M order, which is not a cost model. `η = σ/ADV` says trading one full ADV in a day
costs about one daily volatility of slippage: dimensionally consistent, and it puts a
10%-ADV order at 7.7 bps, which is a number you can argue with.

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

`σ`, average daily volume and the live touch spread are pulled from a **real Deribit
instrument**, and this is the default path — there is no invented book to fall back to
unless you ask for one. Temporary impact scales as `σ/ADV`; `ε` is half the quoted
spread, which used to be hardcoded at 0.0625, a US equity tick from before
decimalisation.

Urgency is specified as a **half-life in days** rather than as `λ`. `λ` divides through
by `σ²` and `η̃`, so it carries the instrument's price and volume units: the old default
of `2e-6` was reasonable on a made-up book and puts the half-life at 0.02 days on a real
BTC calibration, i.e. liquidate immediately. A half-life is the same number whatever you
are trading.

```bash
uv run python scripts/run_execution.py                     # real Deribit calibration
uv run python scripts/run_execution.py --participation 0.25 --half-life 0.5
uv run python scripts/run_execution.py --offline           # illustrative, no network
uv run pytest                                              # trajectory + frontier invariants
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
