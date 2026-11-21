# Registered predictions

These are the prediction sections of the author's private, dated planning log for this project, copied verbatim. Each section was written before the runs it concerns, as its heading states. The outcomes were recorded afterwards in the same log and are reported against these criteria in Supplementary Table S5 of the paper. The log was not deposited in a public registry, so the dates are as written at the time.

## Predictions registered before batch 3 (11 Nov 2026)
- **P1, staleness explains failed inheritance.** If inherited notes fail mainly because they go stale before the newborn can read them, then:
  - in a frozen world, an inherited notebook beats a wiped one for newborns by more than 0.02;
  - with drift every 8,000 steps, the gap is smaller but still positive;
  - the same holds for kept versus erased pages after a mass die-off.
  If the gaps stay within ±0.02 in a frozen world, inheritance fails for reasons other than staleness.
- **P2, honesty threshold.**
  - With own pages readable, self-use pays a little for honesty, so the switch from mostly honest to mostly false writes lies at a negative credit: honest writes above 50% at c = −0.02, below 50% at c = −0.1 (already 4%).
  - With own pages hidden, the switch moves to c ≈ 0: above 50% at c = +0.02, below 50% at c = −0.02.
- **P3, bottlenecks.** Newborn reward of the cohort reborn at a bottleneck rises with the number of experienced survivors. With all agents dead it matches the mass die-off (about 0.59). With half surviving it is within 0.02 of staggered deaths (0.72).

## Model v1 (11 Nov 2026, written before batch 3 results)
1. **How fresh the record is.**
   - After a machine changes, its wiki majority stays wrong until half the authors have rewritten it correctly.
   - An author revisits a given machine every K·V steps on average, and a rewrite is correct with probability q (the share of correct writes).
   - So each change leaves the majority wrong for about K·V·ln(1/(1 − 1/(2q))) steps, and the share of wrong majorities is about (K·V/D)·ln(1/(1 − 1/(2q))) + ε.
   - Using the measured q and one offset (ε = 0.006), this predicts the share of wrong majorities in 14 existing settings within ±0.03. Those settings vary drift (500–8,000), machines (4–32), agents (4–32) and lifespan (100–900); the worst case is 4 agents (+0.03).
   - Drift, the number of machines and visit length matter only through K·V/D: changes per author revisit.
   - **Prediction for batch 3:** `cited_static` majority right ≈ 0.99.
2. **The heir erases the inheritance before it can read it.**
   - A newborn writes on about 10% of steps (write prior), half of them false arms. Per visit, P(no write) = 0.9⁸ ≈ 0.43, and each machine is visited about every 2 updates.
   - So about 88% of the inherited entries are overwritten within 5 updates. A newborn needs about 100 updates to use a notebook, and about 25 with kept features.
   - **Prediction for batch 3:** inherited minus wiped notebook stays within ±0.02 even in a frozen world, for fresh and kept-feature heirs. P1's staleness hypothesis fails. The die-off gap (pages kept minus erased) stays small (≤ 0.05) at every drift rate.
   - **Prediction for batch 4 (juvenile silence: no writing during the first 50 updates of a life):**
     - inherited minus wiped notebook exceeds 0.02 in a frozen world, and the gap grows with how long facts last;
     - the die-off gap also grows.
3. **Two supports for honesty.**
   - Write the incentive as self-use S + credit c·C − competition ρ·H. The record is mostly honest when it is positive.
   - The signs of all six existing cases match:
     - individual 77% honest;
     - hidden page 52%;
     - competition with own page 73%;
     - hidden page plus competition 24%;
     - credit at least 95% in every case;
     - rivalry at most 4%.
   - A smooth logistic version of this model fails. Fitted to individual and cited, it predicts 66% honest at c = −0.1, but 4% was observed. The switch is far sharper than linear incentives allow, which suggests positive feedback between readers' trust and writers' incentives (two stable regimes).
   - **Prediction for batch 3:**
     - with readable pages the switch lies between −0.1 and 0;
     - with hidden pages it lies at about 0;
     - in both cases honest writes jump from below 20% to above 80% within one step of the credit grid.

## Predictions registered before batch 5, evolution of the juvenile period (11 Nov 2026, evening)
Each agent in each world copy carries a heritable juvenile period. At death, the heir copies the juvenile period of the fitter of two random copies (same slot) and mutates it (σ = 10 updates). Twelve lifetimes, starting from 0 or from 150 updates.
- **E1:** private notebooks, frozen world. A long juvenile period (at least 50 updates) evolves from 0, and the population started at 150 stays at 50 or more.
- **E2:** private notebooks, drift every 2,000 steps. The final juvenile period is shorter than in E1, because silent juveniles cannot keep their own notes current.
- **E3:** heirs keep learned features, frozen world. The juvenile period rises fastest from 0 (largest individual benefit, +0.31).
- **E4:** shared wiki kept by staggered contemporaries. The juvenile period stays short: below 25 updates from 0, and it falls from 150.
- **E5:** mass die-offs in a frozen world. The juvenile period evolves further under group selection (whole copies reproduce) than under individual selection. One agent's silence mostly protects a record that others read, so it is largely a public good.
- **Dose:** in a frozen world, the value of inheritance (inherited minus erased) rises with the fixed juvenile period from 0 to about 50 updates, then levels off or falls by 100.
- **Clarification added before the runs, after the CPU check.** Mutation plus the floor at 0 makes a juvenile period that starts at 0 drift upwards with no selection at all: about 22 updates after 11 deaths. So "evolves" in E1–E5 means a departure from this neutral expectation, simulated with random parents. The analysis reports it as the "selection effect" with a 95% CI across seeds.

## Predictions registered before batch 6, 36-lifetime evolution (12 Nov 2026)
Settings: notebooks with facts changing every 2,000 or 8,000 steps, fresh heirs or heirs that keep features, each starting from 0 and from 150 updates.
- **L1, convergence:** in each setting, the populations started at 0 and at 150 end less than 50 updates apart (they started 150 apart), approaching a common juvenile period.
- **L2, facts that last:** that common period is longer with drift every 8,000 steps than with 2,000.
- **L3, early readers:** it is longer for heirs that keep features than for fresh heirs.
- **L4:** for fresh heirs with drift every 2,000 steps it is short (under 30 updates).

## Predictions registered before batch 7, the review's tests (12 Nov 2026, 16:40; written before any code or runs)
Prompted by an independent review: does the inheritance failure come from one setting (write prior 0.9), is the evolved juvenile period a real optimum or mutation–selection balance near zero, and is the smooth honesty curve hiding two stable regimes?

**K, write prior.** Newborns' initial probability of keeping an entry raised from 0.9 to 0.97 and 0.99 (write prior 0.03 and 0.01), no juvenile period. Frozen world for the inheritance pairs; shared wiki at the default drift.
- K1: inherited minus erased newborn reward rises with the keep prior (0.9 < 0.97 < 0.99) for the private notebook, the notebook with kept features, and the pages after a mass die-off.
- K2: at 0.99 it stays below the gain from a 50-update juvenile period (+0.18, +0.31, +0.50). Expected size at 0.99: under +0.05 for fresh notebook heirs (they need about 100 updates to read), about +0.1 to +0.2 for heirs with kept features, about +0.1 to +0.3 for die-off pages.
- K3: shared wiki with staggered deaths: newborn reward, adult reward and honest writes each within ±0.02 of the 0.9 prior.

**F, fixed juvenile periods.** J fixed at 0, 10, 25, 50 or 100 updates in the four 36-lifetime settings (fresh heirs or kept features; facts changing every 2,000 or 8,000 steps); 6 lifetimes. Fitness = mean reward per pull over completed lives in the second half of the run.
- F1: in every setting the best fixed J is above zero, and its fitness exceeds J = 0 by more than the 95% CI.
- F2: the best fixed J is at least as long with facts changing every 8,000 steps as every 2,000.
- F3: with facts changing every 2,000 steps, the best fixed J is longer than the evolved one (14–24). Selection on a juvenile period that differs between copies of a shared learner holds it below the period that is best when every copy has it. If F1 fails (J = 0 best), the evolved values are mutation–selection balance near zero, as the reviewer suggested.

**H, hysteresis near the honesty threshold.** The credit is +0.5 (honest start) or −0.5 (lying start) for the first 3 lifetimes, then switches to the target for 6 more. Targets: own page readable c = −0.05, −0.02, 0; hidden c = −0.02, 0, +0.02. Measured over the last third.
- H1: for each target, honest writes differ by less than 0.05 between the two starts (no hysteresis).
- H2: within each run, the honest-write share across the 256 world copies is unimodal, with a copy-to-copy SD below 0.1.
- H3: final honest writes are within 0.05 of the constant-credit runs at the same c.

## Predictions registered before batch 8, second round of review tests (12 Nov 2026, 19:15; written before any code or runs)
Prompted by two further reviews (Claude 7/10, ChatGPT 7.5/10). Three seeds unless stated.

**R, where liars sit.** Today deceivers occupy the same seats (the last eight) in all 256 copies, and redrawn names use the same permutation in every copy. New controls: own entry shown first with the others in a fixed order shared by all copies ("own first, fixed"); seats drawn at random for each copy and fixed within it ("seats per copy"); the same, redrawn once a lifetime independently in each copy; and seats per copy without liars.
- R1: with 8 coordinated deceivers and seats per copy, newborn reward is within ±0.05 of anonymous authors (0.317) and more than 0.2 below "own first, fixed". A memoryless reader cannot hold a per-copy reputation, so named seats protect only because trust is stored in weights shared by all copies.
- R2: "own first, fixed" is within ±0.05 of the current fixed named authors (0.628).
- R3: without liars, seats per copy cost nothing: within ±0.02 of the shared wiki with credit (0.719).

**I, history test with learners that never die.** The 12 switch settings of batch 7 repeated with no deaths.
- I1: honest writes from the two starts end less than 0.05 apart (point estimates) for every target.
- I2: convergence after the switch (smoothed honest writes first within 0.05 of their final level) takes longer than with deaths in at least 4 of the 6 targets.

**G, group selection in the shared wiki (36 lifetimes, from 0 and from 150; individual and group selection).**
- G1: from 150, group selection ends at a shorter juvenile period than individual selection, with non-overlapping 95% CIs.
- G2: from 0, group selection gives a negative selection effect (CI excluding zero); individual selection is near neutral (CI including zero).
- G3: from 150, newborn reward over the last third is higher under group than under individual selection.

**F (continued), a finer and wider grid of fixed juvenile periods.** New lengths: 15 and 35 (fresh heirs, drift 2,000); 5 and 15 (kept features, drift 2,000); 150 and 200 (fresh heirs, drift 8,000); 15 and 35 (kept features, drift 8,000).
- F4: for fresh heirs with drift 8,000, fitness peaks at 100 updates or later (fitness at 150 or 200 at least that at 100 minus 0.005).
- F5: with drift 2,000, the evolved lengths (22–24 for fresh heirs, 14–15 with kept features) lie inside the range of fixed lengths whose fitness is within 0.005 of the best.

**M, agents with a memory of their own.** Each agent keeps, for every machine, its win and pull counts from its own past visits (halved at each new visit to that machine, erased at death) and sees them in its observation.
- M1: adults without a record reach at least 0.75 (the Bayesian own-memory reference is 0.87).
- M2: a shared wiki still raises newborn reward by at least 0.2 over no record.
- M3: the wiki's advantage for adults falls below 0.05 (it is 0.28 without own memory).
- M4: in a frozen world, inherited minus erased notebook is within ±0.02 without a juvenile period and above +0.05 with a 50-update juvenile period.

**Seeds.** Seeds 3–9 are added (10 in total) for the registered comparisons: the 12 history settings and their 6 constant-credit references; the inheritance pairs with and without a juvenile period and at keep 0.99; the shared wiki with and without a juvenile period and at keep 0.99; the fixed juvenile periods with drift 2,000; and the four identity settings with 8 deceivers. Batch 7's H1 and H3 are re-judged with 10 seeds by the same criteria.

**Statistics, decided now.** Verdicts become exclusive. "Equivalent" means the 90% interval lies inside ±m (two one-sided tests at 5%), split into "equivalent" (the 95% interval includes zero) and "small but non-zero" (it does not). "Larger than m" means the 95% interval lies beyond ±m. "Different" means the 95% interval excludes zero otherwise. Anything else is "inconclusive". The margin is m = 0.02 for rewards, about 4% of the gap between no record (0.315) and a perfect record (0.788) for newborns, and m = 0.05 for the registered honesty tests.

## Prediction registered before the adult-only re-analysis (13 Nov 2026, 12:10; no new runs, written before the analysis code)
Mortal history runs (batch 7, 10 seeds), honest writes over the last third counted only for agents older than 50 updates (adults); newborns' writes are reported separately.
- A1: adults' honest writes after an honest and after a lying start are equivalent within ±0.05 (90% interval) in at least 5 of the 6 targets.
- A2: adults are at least as honest as newborns at every target.

## Predictions registered before batch 9, turnover and the lock (13 Nov 2026, 12:58; written before any code or runs)
All runs: the credit switch of batch 7 (+0.5 or −0.5 until update 900, then the target), 2,700 updates in total, 10 seeds unless stated. Honest writes are measured over the last third (updates 1,800–2,700), all ages unless stated.

**I, more seeds.** Seeds 3–9 for the 12 immortal switch settings of batch 8.
- I1 re-judged with 10 seeds: honest start minus lying start lies beyond +0.05 (95% interval) at readable c = −0.02 and 0 and at hidden c = 0 and +0.02. At readable −0.05 and hidden −0.02 the point estimate stays positive.

**D, turnover dose.** Lifespans of 100, 900 and 2,700 updates (27, 3 and 1 lifetimes of the same 2,700-update run), compared with the existing 300 and immortal runs. Targets: own page readable, c = 0; own page hidden, c = +0.02. Both starts.
- D1: the honest-minus-lying gap grows with lifespan. Point estimates are ordered 300 < 900 < 2,700 < never at both targets, and 100 is no larger than 300 + 0.02.
- D2: at lifespans 100 and 300 the gap is equivalent within ±0.05 (90% interval); at 2,700 it exceeds +0.2.
- D3: recovery time grows with lifespan. Recovery time is the number of updates after the switch until the lying start's honest writes, smoothed over 25 updates, first exceed 0.35 (a run that never does counts as 1,800). Seed means are ordered 100 < 300 < 900 < 2,700, and 900 is at least 1.5 times 300.

**X, which side holds the lock.** Immortal, lying start, same two targets. At the switch, every agent's pull head (the reader), write head (the writer), both heads (hidden layers kept), or whole network is re-initialised once, with its optimiser state.
- X1, readers only: the population stays lying (honest writes below 0.2 at both targets). Fresh readers learn again, from a record that still lies, to avoid the arms it names.
- X2, writers only: the population escapes (honest writes at least 0.5 at both targets). Fresh write heads write honestly about half the time, so the plurality of entries soon names the good arm; readers relearn to follow it faster than writers relearn to lie.
- X3, both heads: escapes (at least 0.5).
- X4, whole network: escapes and ends within 0.1 of the mortal lying-start level (0.776 readable c = 0; 0.729 hidden +0.02).
- Reading prepared in advance: if X1 and X2 hold, the lock is held by what the writers keep writing, and readers only respond to it; turnover breaks it by bringing in writers without the convention.

## Predictions registered before batch 10, the fourth review's tests (13 Nov 2026, 18:46; written before any code or runs)
Prompted by the fourth review (Claude, 7/10). Credit-switch runs as in batch 7 (+0.5 or −0.5 until update 900, then the target; 2,700 updates). "Honest writes" over the last third, all ages, all writes the agents' policies make. 10 seeds unless stated.

**N, random writes without death.** Immortal learners whose sampled write action is replaced, with probability ε per step, by *report belief* or *report a false arm* (equally likely) and trained on as their own action, as a newborn's random writes are. ε = 0.02 (about the population rate of newborns' writes) and 0.10 (a newborn's own rate). Targets: own pages hidden, c = +0.02 (honesty paid by others' reuse) and own pages readable, c = 0 (paid by self-use only).
- N1: hidden +0.02, lying start: ε = 0.10 ends at ≥ 0.5 honest writes; ε = 0.02 ends above the noiseless immortal lying start (0.301) with the 95% interval of the difference above +0.05.
- N2: readable c = 0, lying start: stays below 0.25 at both ε (the floor from random writes alone is about 0.14 at ε = 0.10).
- N3: honest start, ε = 0.10: stays above 0.75 at both targets.
- Reading prepared: if N1 and N2 hold, random writing suffices where others' reuse pays, but where only self-use pays a fresh reader is also needed, which only replacement supplies. If N2 fails, "mortality keeps honesty current" narrows to "variation in writing keeps honesty current".

**S, strong credit.** Lying start switched to c = +0.5, own pages readable or hidden, immortal and mortal.
- S1: mortal populations end at ≥ 0.9 honest writes at both.
- S2: immortal, own pages readable: stays below 0.25 (readers avoid named arms, so the credit is almost never paid, and the author's own reader punishes the truth).
- S3: immortal, own pages hidden: ends above the +0.02 level (0.301) with the 95% interval of the difference above +0.05 (the lock leaks through the few readers who follow, and a larger credit amplifies the leak).

**J, silence against renewal.** Mortal (lives of 300), both starts, juvenile periods of 50 and 150 updates, at the two targets of N.
- J1: recovery time of the lying start (updates after the switch until honest writes, smoothed over 25 updates, first exceed 0.35) is ordered J = 0 < 50 < 150 at both targets (J = 0: 118 and 122), and J = 150 exceeds J = 0 by at least 100 updates.
- J2: over the last third, honest and lying starts are equivalent within ±0.05 (90% interval) at J = 50 and J = 150.
- J3: with an honest start, honest writes at J = 150 are at least those at J = 0 minus 0.02 (silence removes newborns' near-random writes).
- Reading prepared: silence delays renewal by about its length; protecting an inheritance costs a slower escape from a bad convention.

**R, rivalry that destroys information.** An author loses c times its share of every win by a reader who saw its entry, whatever arm the reader pulled (c = −0.5), own pages readable or hidden; compared with the existing rivalry (an author loses only when a reader pulls the arm its entry names and wins; c = −0.5, readable: information 0.22, adults 0.721) and the same named rivalry with own pages hidden (new setting).
- R1: information (reward lost by final agents reading another copy's wiki) falls below 0.05 with own pages hidden and below 0.10 with them readable; with named rivalry and own pages hidden it stays at least 0.15.
- R2: own pages hidden: newborns within ±0.03 and adults within ±0.05 of a world without a record (0.315 and 0.585).

**T, a forward test of the timescale rule** (3 seeds). Private notebook, fresh heirs: learning rate ×0.5 and ×2 in a frozen world, and the default learning rate with facts changing every 8,000 steps; juvenile periods 0, 10, 25, 50 and 100 updates, inherited and erased. Reading time and value share as in the paper (reading time = updates to exceed 0.61 with J = 50 and the record inherited; share relative to J = 100).
- T1: reading time is ordered learning rate ×0.5 > default (41) > ×2.
- T2: for J = 0, 10, 25 and 50 in the three conditions (12 points), the share predicted by the fixed curve (half value at ratio 1.10, slope 1.50 in log ratio) is within ±0.20 of the observed share in at least 9 of 12 points, with a mean absolute error of at most 0.15.

**C, fewer copies** (3 seeds). 64 copies of the world instead of 256.
- C1: the shared wiki with credit gives newborns at least twice the reward of no record.
- C2: with own pages readable and c = 0, the immortal lying start ends below 0.25 honest writes and the mortal lying start above 0.5.


## Predictions registered before batch 11, corrections after the independent review (16 Nov 2026, 23:45; written before any runs)
The independent review of version 5 found two implementation problems. (1) Group selection in the shared wiki (`evo_wiki_group_*_long`, staggered deaths) overwrote the juvenile periods of all sixteen slots whenever agent 0 died, so living adults could be put back into a juvenile period mid-life, and group fitness mixed windows. (2) Credit was computed from the full wiki, so in private-notebook settings an author was paid when another agent won on the arm its unseen page named. The simulator now has two options, both off by default so every earlier result is unchanged: `juvenile_group_birth` (each copy inherits the fitter copy's juvenile periods once per lifetime, judged by mean reward per agent-step since the last selection; each heir takes its slot's inherited value at birth) and `credit_seen_only` (credit and blame go only to authors whose entry the reader could see). Three seeds unless stated.

**G (re-tested), group selection in the shared wiki with the corrected code.** `evo_wiki_groupb_from0_long` and `evo_wiki_groupb_from150_long` (36 lifetimes, credit +0.5), compared with the existing individual-selection runs `evo_wiki_from0_long` and `evo_wiki_from150_long`. The same predictions as G1–G3:
- G1b: from 150, group selection ends at a shorter mean juvenile period than individual selection (89.3 [77.5, 101.1]), with non-overlapping 95% intervals.
- G2b: from 0, the group-selection effect (final mean minus the random-parent control, 41.9) is negative, with its 95% interval below zero.
- G3b: from 150, newborn reward over the last third is higher under group than under individual selection (0.665), with the 95% interval of the difference above zero.

**V, private notebooks with credit only for entries readers can see.** Each inheritance pair of Table 4 rerun with `credit_seen_only=True`: fresh heirs with facts changing every 2,000 steps and in a frozen world, each without and with a 50-update juvenile period; heirs with kept features in a frozen world, without and with it (12 settings).
- V1: without a juvenile period, inherited minus erased newborn reward is equivalent within ±0.02 for fresh heirs, at 2,000 steps and in a frozen world.
- V2: with a 50-update juvenile period in a frozen world, inherited minus erased exceeds +0.10 for fresh heirs and +0.20 for heirs with kept features (the 95% interval of the difference lies above these values).
- V3: every inherited-minus-erased value lies within 0.05 of its value under the original credit rule (Table 4).
- V4: honest writes in the private-notebook settings fall below their original 96–99% and lie between 0.60 and 0.90 (self-use alone pays for honesty, as in the shared wiki with own pages readable and c = 0, 77%).

**W, measurement windows (no new runs).** From the saved logs: newborn and adult reward of the mortal credit-switch runs over lives born at least one lifetime after the switch (update 1,200 or later), and newborn reward of the 36-lifetime evolution runs over lives born in the last third.
- W1: after the switch, newborn reward after an honest and after a lying start is equivalent within ±0.02 at no fewer than 5 of the 6 targets.


## Predictions registered before batch 12, why group selection lagged (21 Nov 2026, 11:55; written before any code or runs)
Batch 11: with staggered deaths, group selection once a lifetime (values taken at birth) shortened a 150-update juvenile period less than individual selection (123.7 [103.0, 144.4] against 89.3 [77.5, 101.1]). Two explanations:
- **Frequency:** individual selection acts at every death (16 times a lifetime per copy), group selection once a lifetime, and its choice reaches each agent only at that agent's next birth.
- **Dilution:** a copy's mean reward reflects all sixteen juvenile periods and the copy's luck, so it says little about any one agent's juvenile period.

**Design.** New option `juvenile_group_each_death`: group selection at every death, with the mechanics of individual selection. When agent i dies, each copy's heir in position i copies the juvenile period of position i in the fitter of two random copies and mutates it (σ = 10, as before). Fitness is the copy's mean reward per agent and step over agent i's life (all sixteen agents), instead of agent i's own reward over its life. So frequency, timing and mutation match individual selection exactly; only what selection sees differs. (Copying all sixteen values at every death was rejected: it would mutate each value about 16 times a lifetime.) Settings `evo_wiki_groupd_from0_long` and `evo_wiki_groupd_from150_long`: 36 lifetimes, credit 0.5, three seeds, as the other wiki evolution runs. Differences are Welch 95% intervals over seeds.

- **Y1:** from 150, group selection at every death ends with a shorter mean juvenile period than group selection once a lifetime (123.7): difference below zero, 95% interval excluding zero.
- **Y2:** from 150, group selection at every death does not end longer than individual selection (89.3): the 95% interval of the difference includes zero or lies below it.
- **Y3:** from 150, newborn reward over the last third under group selection at every death is higher than under group selection once a lifetime (0.627), and the 95% interval of its difference from individual selection (0.665) includes zero or lies above it.
- **Y4:** from 0, the selection effect of group selection at every death is negative (95% interval below zero), as for the other two kinds.

**Prepared readings.** Y1 and Y2 hold → frequency explains the lag: judged at every death, group selection does as well as individual selection, and the paper's untested explanation is supported. Y1 fails and the result sits near 124 → dilution: the copy's reward is too weak a signal of one agent's juvenile period, and the paper's explanation is replaced. Y1 holds but Y2 fails → both play a part. With three seeds, intervals of ±20 updates or more are expected, so an inconclusive result is possible and will be reported as such.

