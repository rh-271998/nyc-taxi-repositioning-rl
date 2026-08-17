# Journey — a running log

A chronological log of decisions, dead-ends, bugs, and findings. Kept live from day
one, so it records what was actually thought at each step — including the hypothesis
that turned out wrong.

---

## Aug 7 — Framing the project

Goal: a value-based RL project that "feels the RL" — Bellman/TD from the ground up — on a
**real-world problem with real data**, and solves **one** problem with **three algorithms
progressively** (tabular Q-learning → DQN → Double DQN), each motivated by the previous
one's limitation. Ruled out toy tasks (CartPole/LunarLander) and problems already done
(traffic signals, dynamic pricing).

Chose **NYC taxi driver repositioning**: where should an idle driver go to catch the next
fare? It's intuitive, has obvious real-world weight, and — crucially — the tabular→DQN
story has a natural home: a coarse `(zone, time)` state is table-feasible, but the
ambition to react to richer conditions motivates function approximation.

Data: **NYC TLC yellow-cab trip records** (gold-standard public data). Picked June 2019
(`PULocationID`/`DOLocationID` zone schema, 263 zones). Plan to restrict to Manhattan to
keep demand dense and the zone count manageable.

## Aug 10–12 — M0: building the environment's data (`preprocess.py`)

Aggregated 6.9M trips into per-`(zone, bucket)` lookups. Several decisions and corrections:

- **Time = time-of-day, not absolute timeline.** First cut accidentally ranked distinct
  timestamps, making every 5-min slot ultra-sparse. Fixed to
  `seconds_from_midnight // (5·60)` → buckets `0..287`, computed identically for pickup and
  dropoff.
- **Medians, not means**, for fare and duration (robust to outliers).
- **Fare depends on destination, like duration.** Caught my own inconsistency — metered
  fare scales with distance/time, so it's keyed on `(origin → dest)`, not origin alone.
- **Intra-Manhattan only** (`PU AND DO in Manhattan`). Makes the environment a **closed
  sub-graph** — the driver can never be transported off the map, so every lookup is total.
  Stated openly as a simplification.
- **The zone-adjacency bug.** Building the movement graph with `geopandas` `touches`
  *silently* returned zero neighbors for several landlocked uptown zones (Hamilton Heights,
  Inwood, Washington Heights N). Cause: sub-foot **sliver gaps** in the shapefile polygons,
  which `touches` requires to be perfectly shared. Fix: filter to Manhattan first, then
  `.buffer(10 ft)` + `intersects` — stable across 5–50 ft, closing slivers without bridging
  streets. Only the genuine water-separated islands remain neighborless, so the env must
  allow **stay-only zones**.

Six artifacts saved to `data/artifacts/` so training never re-parses the raw parquet.

## Aug 12 — MDP and environment design

Worked through the modeling choices before writing `env.py`:

- **Event-driven Semi-MDP.** Decisions only when idle; actions advance the clock by their
  own (variable) duration. Chosen over a fixed 5-min step to avoid a "busy-for-N-steps"
  timer in the state. Key consequence: **terminate by the clock, not step count**, so the
  objective is honestly "earnings in a fixed shift" (γ = 1).
- **Action space = 69**, encoding "stay" as "move to your own zone." Absolute (per-target-
  zone) rather than relative (per-neighbor-slot) so the encoding survives unchanged into
  DQN, where a stable action meaning matters for the network.
- **The pickup model** — the piece that makes repositioning a real decision. Demand is a
  *count shared across all drivers*; converted to a per-bucket pickup probability
  `p = 1 − exp(−(demand/30)/κ)`. Two subtleties nailed here: demand is a **monthly total**
  (÷30 for one driver's day), and **κ is a stand-in for competition** — without it a single
  agent gets every fare and repositioning becomes pointless.
- **κ = 5**, chosen by calibration against the real demand distribution (hot ≈ 5-min wait,
  median ≈ 7-min, cold ≈ 56-min — an ~11× spread that makes driving to a hot zone worth it).
- **Move ≠ auto-fish.** A reposition is its own decision that lands the driver *idle* at the
  target — it does not force a pickup there. This preserves multi-hop repositioning and a
  clean value recursion. Reward is **fare only** (finite horizon auto-penalizes idling).

## Aug 13 — M1 env + M2 tabular Q-learning + eval harness

- **`env.py`** built and sanity-checked with a random-legal-policy rollout (terminates at
  1440 min, plausible earnings).
- **Tabular Q-learning** (`(69,288,69)` table, ε-greedy over legal, TD with γ=1). Bugs
  found and fixed along the way: `select_action` was masking a **view** of the Q-table
  (corrupting it) and masking the wrong indices; the ε-decay schedule divided by the episode
  index (oscillating / div-by-zero).
- 500-episode smoke test climbed \$177 → \$467. Scaled to **50k episodes** (~150s): plateaus
  ~\$900 (converged; the ceiling is coverage of cold, low-value cells).
- **Eval harness** (`eval.py`): the objective is per-episode total return, evaluated on a
  **fixed 1000-seed paired** set (every policy sees identical days). Fixed real bugs here
  too — it was averaging *within-episode step rewards* instead of across-episode totals, and
  the paired standard error was computed as `SE₁ − SE₂` (meaningless) instead of the std of
  the elementwise per-seed deltas. Also caught a training bug where every episode used the
  *same* seed (identical day 50k times) → switched to a per-episode seed.
- **M2 result:** greedy **\$922.4 ± 8.8** vs random **\$178.6 ± 2.3** — **+\$744 ± 7**,
  ~5.2×, winning 99.9% of days. Ironclad.

## Aug 16 — M3: DQN

- Built from scratch: `QNet` MLP, `ReplayBuffer`, `TaxiAgentDQN` with online + target nets.
- **State representation** was the real design work. Rejected one-hot (reproduces the
  table's no-generalization problem); used **centroid + cyclic time + linear time** — the
  same information as the table, in a form that lets nearby states share learning.
- Added a `centroids.npy` artifact (aspect-ratio-preserving normalization, so Manhattan's
  tall-thin shape isn't distorted). Fixed a terminal-bucket-288 indexing bug that only
  surfaced once features indexed by bucket.
- **The hypothesis:** DQN generalizes → it should beat tabular's coverage-limited plateau.
- **The result — a null.** DQN **\$903.4 ± 9.2**, i.e. −\$19 *vs* tabular. DQN did **not**
  beat the table. The lesson: at only ~20k states the table is **feasible**, so it's the
  gold standard (exact) and DQN can at best match it — the "coverage ceiling" was actually
  near-optimal. DQN's genuine win is **sample efficiency**: it reached ~\$900 in ~10k
  episodes vs tabular's tens of thousands.

## Aug 17 — Tier-2 ablation, then M4: Double DQN

- **Ablation first.** Hypothesized the −\$19 came from the centroid being a *lossy* proxy
  for zone identity, so I fed DQN the exact demand signal (current + neighbor pickup
  probability, +3 features). **It changed nothing** (\$903.4 → \$903.4). Conclusion: the gap
  is **not** an information problem — it's inherent function-approximation smoothing. A clean
  ablation that ruled out my own hypothesis.
- **Decision point.** With DQN ≈ tabular there's no return headroom left. Rather than force a
  win by re-engineering the problem, chose to keep the honest result and make M4 about
  **overestimation** — the textbook-correct reason Double DQN exists, measurable regardless
  of return.
- **Double DQN**: a one-flag change decoupling action *selection* (online net) from
  *evaluation* (target net). Bug caught: the online-net mask used `batchMask` (float,
  inverted) instead of `batchMask == 0`.
- **The headline result.** Comparing predicted `Q(s₀)` (the network's whole-shift forecast)
  to realized return over 1000 seeds: **DQN overestimates by \$98 ± 5; Double DQN by only
  \$16 ± 4** — an ~84% reduction, ~13-SE gap. The better-calibrated values also nudged the
  return to **\$928.6** (marginally above tabular, within noise; clearly above vanilla DQN).

## Aug 17 — Figures

Three figures for the write-up: the **overestimation** diagnostic (the hero image), the
**return comparison** (all learned policies ≈ 5× random), and a **repositioning flow map** —
Manhattan colored by demand with the greedy policy's chosen move per zone across four times
of day. The map validates the policy visually: the "stay & fish" fraction tracks demand
(27/69 zones at 3 AM → 52/69 at the 6 PM peak), and idle drivers are pulled from cold zones
toward the hot core.

---

## The arc, in one line

**Tabular solves it** (near-optimal on a feasible state space) → **DQN matches it** with
5× less data (the residual gap is approximation cost, ablation-confirmed, not missing
information) → **Double DQN fixes DQN's overestimation** (\$98 → \$16 bias). Each step's
limitation genuinely motivated the next — and where the hypothesis was wrong (DQN "should"
beat tabular), the honest result and the reason behind it turned out to be the more
interesting finding.

## What I'd do differently / next

The one change that would make DQN genuinely *necessary* — and able to beat the table — is
to grow the problem beyond a table's reach: **stochastic, observable demand** (real
day-to-day variation), **finer time resolution**, or a **multi-agent** fleet. That's the
natural continuation.
