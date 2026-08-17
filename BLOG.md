# Can you teach an AI where a taxi should wait for its next fare?

Picture a cab driver who's just dropped someone off. The meter's off, and they face a
small decision that repeats a hundred times a shift: **sit here and wait for a fare, or
drive somewhere they think is busier?** Sit in a dead spot and you burn twenty minutes
earning nothing. Chase a hot neighborhood and you might arrive to find everyone already
has a ride.

Every driver builds an instinct for this over years. I wanted to know: can you *learn*
that instinct from data — and how fancy does the method need to be? So I built it three
ways, on **real New York City taxi data** (6.9 million yellow-cab trips), and let the
computer figure out where to send an idle driver, minute by minute, across a full day in
Manhattan.

## The setup

Think of Manhattan as ~69 neighborhoods. At any moment the driver is in one of them, at
some time of day, and picks one move: **fish here**, or **hop to a neighboring
neighborhood**. Real trips decide everything — how likely a fare appears, where it takes
you, how much it pays, how long it takes. The driver works one shift; the goal is simple:
**make the most money before the day ends.**

I compared four "drivers":

1. **Random** — wanders and fishes with no strategy (the sanity-check floor).
2. **A lookup table** — the classic approach: try things thousands of times and memorize,
   for every neighborhood and time, how good each move turned out.
3. **A neural network ("DQN")** — instead of memorizing every case, it *generalizes* from
   features like location and time of day.
4. **A smarter neural network ("Double DQN")** — same idea, with one fix for a subtle bug
   that neural-net value-learners are known to have.

## What happened

**All three learned drivers made about 5× what the random one did.** Wandering earned a
baseline; the strategies earned roughly five times more by fishing the right places at the
right times and repositioning ahead of demand. That part was the easy win.

Then came the interesting part — and it wasn't the headline I expected.

## The twist: the fancy method didn't beat the simple one

I assumed the neural network would *beat* the lookup table. It didn't. It **tied** it.

Why? Because this problem, it turns out, is *small enough that the simple table can just
solve it.* When you can afford to memorize every situation exactly, memorizing is
unbeatable — a neural network that *approximates* can only match it, never surpass it. I
even tried handing the network extra information to close the tiny gap; it made no
difference, which confirmed the gap wasn't a lack of information — it was the unavoidable
cost of *approximating* instead of memorizing.

So does the neural network buy you nothing? No — its win is on a different axis: it learned
the same quality of driving from **about five times less experience.** The table needed
tens of thousands of practice days; the network got there in a fraction. That's the real
lesson about when you reach for the fancier tool: *not* when the simple one already fits in
memory, but when the problem is too big to memorize — then generalization is the only way.

Being able to say that clearly — "here's exactly where the fancy method helps, and here's
where it doesn't" — is more useful than a forced victory.

## The genuinely cool result: an overconfident forecaster, corrected

Here's where the third driver earned its keep. Neural-net value-learners have a known flaw:
they're **overconfident.** Because of a quirk in how they learn, they systematically
*overestimate* how well they'll do.

I measured it directly. I asked each driver, at the start of the day: *how much do you
think you'll make?* — then compared that to what they actually earned:

- The plain neural network predicted **\$1,002** and earned **\$903**. It overestimated by
  ~\$98 — about 11% too optimistic, every single day.
- The "Double" version predicted **\$944** and earned **\$929** — off by just ~\$16.

The one small fix cut the overconfidence by **~84%.** That's the textbook result, and
seeing it appear cleanly in my own from-scratch build — on real data — was the most
satisfying moment of the project. As a bonus, a driver with honest expectations also made
slightly *better* decisions.

## What the AI actually learned

The best part is you can *watch* the strategy. Coloring the map by how busy each
neighborhood is, and drawing where the AI sends idle drivers:

- **At 3 a.m.**, most of the city is dead, so the AI keeps drivers *moving* — pulling them
  toward the few busy pockets downtown.
- **At the 6 p.m. rush**, demand is everywhere, so it mostly says *stay and fish* — no need
  to chase.
- Throughout, idle drivers get nudged **from quiet areas toward busy ones**, and the target
  shifts as the city wakes up and winds down.

Nobody programmed that. It emerged from the data.

## The honest fine print

This is a clean experiment, not a real dispatch system. It models **one** driver who never
competes with other cabs, and it uses a simplifying knob to stand in for that competition —
so the *dollar amounts are optimistic*. What's trustworthy is the *comparison*: random vs
learned, simple vs fancy, overconfident vs corrected. Those relationships are the real
findings.

## The takeaway

The lesson isn't "AI drives taxis better." It's more useful:

> A simple method, when it fits, is often unbeatable — and knowing *when it stops fitting*
> is the actual skill. The fancier tool earns its place when the problem outgrows
> memorization, or when you need honest, well-calibrated confidence.

Sometimes the most valuable result of an experiment is a precise map of where the simple
answer is good enough — and this one drew that map.
