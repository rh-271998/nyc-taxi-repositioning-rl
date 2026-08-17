# Technical write-up — NYC Taxi Repositioning with value-based RL

This document is the methodology deep-dive: the MDP formulation, how the environment
is estimated from real data, the pickup model that makes repositioning meaningful, the
three algorithms, and the evaluation. It is written to stand alone.

---

## 1. Problem formulation

A single idle taxi driver repositions across Manhattan to maximize earnings over a
shift. We model it as a **finite-horizon Semi-Markov Decision Process (SMDP)**.

- **State** `s = (zone, t)`: one of 69 Manhattan taxi zones and a time-of-day bucket
  `t ∈ {0..287}` (five-minute buckets over a 24-hour day).
- **Action** `a`: an integer in `0..68` naming a **target zone**. Two cases:
  - `a == zone` → **stay and fish** for a fare in the current zone.
  - `a ∈ neighbors(zone)` → **reposition** (drive empty) to an adjacent zone.
  - All other actions are **illegal** and masked out. Encoding "stay" as "move to your
    own zone" keeps the action space exactly the size of the zone set.
- **Time is continuous** internally (a clock in minutes, `0 → 1440`); `t` is derived as
  `⌊clock / 5⌋`. The clock is the single source of truth, so the time bucket can never
  drift out of sync.
- **Horizon:** the episode ends when `clock ≥ 1440` (one full day). **γ = 1** — the
  undiscounted return is therefore *exactly the total earnings for the shift*, the most
  interpretable possible objective.
- **Reward:** the fare collected on a completed trip (moves earn nothing; an optional,
  currently-zero move cost is supported).

### Why a Semi-MDP, not a fixed-step MDP

Actions take **different amounts of real time**: a reposition to a neighbor is a few
minutes; serving a fare can be 20+ minutes. A fixed 5-minute step would force a
"busy-for-N-more-steps" timer into the state. Instead, **decisions happen only when the
driver is idle**, and each action advances the clock by its *own* duration. The one
correctness consequence: **the episode must terminate by the clock, not by a step count**
— otherwise a policy could farm many tiny trips. Terminating on `clock ≥ 1440` with γ=1
makes "maximize return" identical to "maximize earnings in the fixed shift."

---

## 2. Data pipeline (`preprocess.py`)

Everything the environment needs is estimated **once** from 6.9M real yellow-cab trips
(NYC TLC, June 2019) plus the official taxi-zone shapefile, and saved to
`data/artifacts/`. Scope is restricted to **intra-Manhattan** trips (both pickup and
dropoff in Manhattan) so the environment is a **closed sub-graph** — a driver can never be
transported off the map, which keeps every lookup total.

| Artifact | Shape | Meaning |
|---|---|---|
| `demand.npy` | `(69, 288)` | trip originations per `(zone, bucket)` — a **monthly total** |
| `destinationProbab.npy` | `(69, 288)` object | for each `(zone, bucket)`: `(dest_idx[], prob[])`, a distribution over destinations |
| `fare_duration.npy` | `(69, 69)` object | per `(origin, dest)`: `(median fare, median duration_minutes)` |
| `adjacent_zones.json` | dict | zone → list of adjacent zones |
| `centroids.npy` | `(69, 2)` | aspect-ratio-preserving normalized zone centroids |
| `id2idx / idx2id` | dicts | map raw TLC `LocationID` ↔ dense `0..68` index |

Two data details worth calling out:

**Fare/duration fallback ladder.** `odStats` only has O–D pairs that actually occurred,
some with too few trips to trust. For a queried pair we use its median if
`n_trips ≥ 10`; else a **trip-count-weighted average over the destination's neighbors**
(excluding the origin, so an `A→A` zero-distance hop can't corrupt it); else a global
median. This gives every cell a sane value without inventing precision.

**Zone adjacency — a shapefile gotcha.** A naive `geopandas` `touches` predicate
*silently dropped* the real neighbors of several landlocked uptown zones (Hamilton
Heights, Inwood, Washington Heights N) because the polygons have sub-foot **sliver gaps**.
Fixing it required a small buffer: filter to Manhattan first, `.buffer(10 ft)` (the
shapefile CRS is EPSG:2263, in feet), then `intersects`. The result is stable from 5–50 ft
(closes slivers without bridging streets), leaving only the genuinely water-separated
islands (Governor's/Ellis/Liberty, Marble Hill, Roosevelt) with no neighbors. The
environment therefore must tolerate **stay-only zones**.

---

## 3. The pickup model — what makes repositioning meaningful

When the driver **stays**, do they get a fare, and how long do they wait? This is the
environment's most important modeling choice, because it's what makes "stay vs move" a
real decision.

The demand data is a *count of trips shared across all drivers* — not one driver's odds.
We convert it to a **per-bucket pickup probability**:

```
λ = demand[zone, t] / NDAYS          # NDAYS = 30 → per-day intensity a driver sees
p = 1 − exp( −λ / κ )                # probability of a pickup in this 5-min bucket
```

Staying is then a Bernoulli(`p`) each bucket: on success, sample a destination from
`destinationProbab`, collect the fare, advance the clock by the trip's duration, and land
at the destination; on failure, wait one bucket (5 min) and re-decide. Repeated Bernoulli
trials make the **expected wait ≈ 5/p minutes** emerge naturally.

### Two subtleties that this model gets right

1. **The demand is a monthly total.** A single driver on one day sees `demand/30`, not the
   raw count — hence the `/NDAYS`. Feeding the raw count would make every zone look
   infinitely hot.
2. **κ is a stand-in for competition.** With a *single* agent and no supply model, "you
   get the next trip" would make every non-empty zone equally good and **repositioning
   pointless**. κ encodes "how much demand it takes to be reliably matched" — the implicit
   competing-driver density. It is a **design constant, not a fitted parameter** (there is
   no wait-time ground truth to fit). It's chosen by *calibration*: slide the `1−exp` curve
   so the per-day demand distribution maps to realistic waits.

### Calibrating κ

Per-day demand quantiles (over all non-empty cells): p10 ≈ 0.5, median ≈ 6.8, p90 ≈ 27
trips / 5 min. Sweeping κ and reading off expected waits:

| κ | cold (p10) | median | hot (p90) |
|---|---|---|---|
| 1 | 13 min | 5 min | 5 min (top saturates — no gradient) |
| **5** | **~56 min** | **~7 min** | **~5 min** |
| 8 | 88 min | 9 min | 5 min (median over-punished) |

**κ = 5** gives an ~11× hot-vs-cold spread across the *bottom half* of zones — enough
contrast that driving to a hotter zone is worth the travel time, without over-punishing
ordinary zones or saturating the top. That contrast is the entire learning signal.

---

## 4. The environment (`env.py`)

`step(action)` dispatches on the action:

- **Stay (`action == zone`):** run the Bernoulli(`p`) pickup above.
- **Move (`action ∈ neighbors`):** reward 0 (empty drive), advance the clock by the O–D
  travel time, arrive **idle** at the target — *no forced pickup*. This is deliberate:
  keeping the move a separate decision lets the agent **chain multiple hops** toward a hot
  district and re-decide on arrival, and it makes the value recursion clean
  (`Q(s, move→B) = −cost + V(B, t+travel)`).

Reward is **fare only**. No explicit wait penalty is needed: because the horizon is finite
and γ=1, every idle minute silently burns shift time that could have earned — the
opportunity cost is baked in.

Action legality is exposed as a mask; the environment trusts the agent to pass a legal
action (asserted in development). The terminal state's bucket is clamped to `287` so no
downstream array indexed by bucket can go out of range.

---

## 5. M2 — Tabular Q-learning

A `(69, 288, 69)` Q-table (~1.4M cells), ε-greedy over legal actions (illegal masked to
−∞), standard TD update:

```
target = r                                  if done
       = r + γ · maxₐ' Q[zone', t', a']      otherwise   (γ = 1, max over LEGAL a')
Q[s, a] += α · (target − Q[s, a])
```

α = 0.1; ε decays linearly 1.0 → 0.05 over 60% of training; one episode = one full day
with a random start zone (fresh RNG each episode for coverage).

**Result:** trained 50k episodes (~150s), plateaus around \$900 training return.
Greedy eval: **\$922.4 ± 8.8**, ~5.2× the random baseline. The plateau is **convergence**:
with ~120k legal state-action pairs and ~200 updates/episode, coverage is the ceiling —
but the under-visited cells are cold (4 AM, low-demand) states that barely affect return,
so the table is effectively **near-optimal**. On a state space this small, the table is
the gold standard.

---

## 6. M3 — DQN

Replace the table with a small MLP that **generalizes across similar states**.

- **Network:** `QNet`, `inDim → 128 → 128 → 69` (ReLU, raw linear output — Q-values are
  unbounded). Outputs 69 action-values; illegal ones masked before argmax.
- **State features (the key design):** a one-hot `(zone, t)` would just reproduce the
  table's no-generalization problem. Instead, continuous features that encode *similarity*:
  - normalized zone **centroid** `(x, y)` — nearby zones sit close in feature space;
  - **cyclic time** `sin(2π t/288), cos(2π t/288)` — 6:00 and 6:05 are neighbors, midnight
    wraps smoothly;
  - **linear time** `t/288` — a monotonic handle on the finite-horizon "time remaining."
  - *(Tier-2 ablation)* current + neighbor **pickup probability** `p` (self-normalizing in
    `[0,1]`).
- **Stabilizers:** an experience **replay buffer** (breaks temporal correlation) and a
  periodically-synced **target network** (a frozen bootstrap target every 1000 learn-steps).
- **Loss:** MSE between `Q_online(s,a)` (via `gather`) and the target computed under
  `no_grad` from the target net, with `(1−done)` masking the bootstrap. γ = 1 to match the
  tabular objective exactly (so the comparison is apples-to-apples). Adam, lr 1e-3, gradient
  clip 10, batch 64.

**Result:** **\$903.4 ± 9.2** — statistically matches tabular. DQN reaches ~\$900 in
~**10k** episodes vs tabular's tens of thousands: its advantage is **sample efficiency**,
generalizing from far less experience.

### Ablation — is the −\$19 vs tabular an information gap?

No. Feeding DQN the exact demand signal (current + neighbor `p`) as extra features left
the result **unchanged** (\$903.4 → \$903.4, within noise). So the small gap is **inherent
function-approximation cost** — a smooth network can't perfectly replicate the table's
sharp per-cell values — not missing information. This is the clean, honest reason DQN
can't beat a feasible table: the problem isn't information-limited.

---

## 7. M4 — Double DQN

DQN's target `r + γ·maxₐ' Q_target(s', a')` uses the **same network to both select and
evaluate** the best next action. Because Q-estimates are noisy, `max` systematically picks
overestimated values → an **upward bias** that compounds through bootstrapping.

**Double DQN** decouples the two roles:

```
a* = argmaxₐ' Q_online(s', a')      # ONLINE net selects (over legal actions)
target = r + γ · Q_target(s', a*)    # TARGET net evaluates that choice
```

Independent noise between the two nets means the selected action is no longer the one the
evaluator overrates. Implemented as a single `double` flag on the agent, so DQN and Double
DQN are byte-for-byte identical except this target rule — a clean isolation of the effect.

### Measuring overestimation

`Q(s₀) = maxₐ Q(s₀, a)` is the network's **forecast of whole-shift earnings** from the
start state (γ=1, so a single value already spans the whole episode). Comparing it to the
**realized** greedy return over 1000 seeds — the rollout noise averages out — isolates the
bias:

| agent | predicted `Q(s₀)` | realized return | **bias** |
|---|---|---|---|
| DQN | \$1001.6 | \$903.4 | **+\$98.1 ± 4.8** |
| Double DQN | \$944.1 | \$928.6 | **+\$15.6 ± 4.0** |

An **~84% reduction** in overestimation (a ~13-SE gap — ironclad), exactly the textbook
signature. The better-calibrated values also improve the greedy policy modestly
(\$903 → \$929).

---

## 8. Evaluation methodology (`eval.py`)

- **Objective:** total shift earnings, so each episode contributes one scalar (its return).
- **Fixed, paired seeds.** Every policy is evaluated on the **same 1000 seeds**;
  `reset(seed=s)` makes each policy face the *identical* start conditions for seed `s`.
  This cancels the dominant shared variance (which start zone you drew), so a **paired**
  delta `mean(learnedᵢ − randomᵢ)` has a much tighter standard error than comparing two
  independent means. (It cancels the start-condition variance; pickup luck diverges once
  actions differ — standard for a stochastic MDP.)
- **Reported:** per-policy mean / median / std / **standard error** (`std/√N`), the paired
  delta `± SE`, and a **win-rate** (fraction of days the learned policy wins). Win-rate uses
  `≥`, so ~7% of days tie — these are **island starts** where both policies are stuck with
  only "stay" and earn ~\$0, a nice consistency check on the metric.
- **Overestimation diagnostic:** predicted `Q(s₀)` vs realized return per seed (§7).

---

## 9. Limitations

- **Single agent, exogenous demand** — no depletion, no competition; a fleet is a
  multi-agent equilibrium.
- **Fixed κ** stands in for driver supply (no supply data exists in TLC records). Real
  supply chases demand, so the true hot-vs-cold advantage is smaller than modeled — the
  **absolute dollar figures are optimistic**. The **relative** comparisons are the
  trustworthy results.
- **Demand is a monthly average**, deterministic given `(zone, t)` — which is precisely why
  the table is feasible and DQN cannot beat it. Genuine stochastic, *observable* demand is
  the change that would make function approximation strictly necessary.
- **Empty-drive time** is approximated by loaded-trip duration; **fares** use medians and a
  neighbor fallback for thin O–D pairs.

## 10. What would change the conclusions

The honest one-line summary is *"on a tabular-feasible problem, the table wins by being
exact; DQN's value is sample efficiency and scaling, and Double DQN's is calibration."* To
make DQN genuinely necessary — and let it *beat* the table — the problem itself must grow
beyond a table's reach: **stochastic observable demand**, **finer time resolution**, or a
**multi-agent** formulation. Those are the natural next steps.
