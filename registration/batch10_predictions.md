# WikiBandits batch 10: registered predictions

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

## Outcomes of batch 10, the fourth review's tests (13 Nov 2026, 23:52; 53 new settings, 306 runs; totals now 315 settings, 1,687 runs, about 24 GPU-hours)

| Prediction | Outcome |
|---|---|
| N1: random writes break the lock where credit pays (hidden +0.02) | **Failed, strongly.** ε = 0.10: 0.051 honest; ε = 0.02: 0.016, *below* the noiseless 0.301 (−0.284 [−0.345, −0.223]). Courses flat for six lifetimes. |
| N2: random writes do not break it with self-use only (readable 0) | **Held:** 0.053 and 0.056. |
| N3: honest immortals with ε = 0.10 stay above 0.75 | **Held:** 0.949 and 0.950. |
| S1: mortal lying start at c = +0.5 ends ≥ 0.9 | **Held:** 0.979 and 0.978 (within one lifetime). |
| S2: immortal, readable, c = +0.5 stays below 0.25 | **Failed:** 0.651 [0.563, 0.739], still rising (0.74 at the end). |
| S3: immortal, hidden, c = +0.5 ends above the +0.02 level | **Held:** 0.761 vs 0.301 (+0.460 [+0.361, +0.559]), still rising (0.84 at the end). |
| J1: silence delays recovery (0 < 50 < 150, 150 at least +100) | **Failed, reversed at 150:** readable 118, 158, 96 updates; hidden 122, 115, 76. |
| J2: histories still equivalent at J = 50 and 150 | **Held** at both targets. |
| J3: silence does not lower an honest start's honesty | **Failed:** J 150 0.697 vs 0.771 (readable), 0.609 vs 0.730 (hidden); population reward also falls (0.651 vs 0.807). |
| R1: rivalry over any win destroys information | **Failed:** information 0.136 (hidden) and 0.524 (readable); named rivalry with hidden pages keeps 0.216 (that part held). |
| R2: hidden, any-win rivalry: newborns and adults at the no-record level | **Held:** 0.301 and 0.591. |
| T1: reading time ordered by learning rate | **Held:** 150, 41, 18 updates. |
| T2: fixed timescale curve predicts 9 of 12 new points within ±0.20 | **Failed:** 7 of 12, mean error 0.204. Shares sit above the curve for slow learners. |
| C1: 64 copies, wiki doubles newborn reward | **Held:** 0.642 vs 0.283. |
| C2: 64 copies, immortal liars stay locked, mortal ones recover | **Held:** 0.037 vs 0.713. |

**Reading:**
- **Replacement, not variation.** Random writes at a newborn's rate never broke the lock, and with credit they even stopped the slow recovery seen without them. Learners that hold the convention repair it: their write rate rose from 0.10 to 0.63–0.91 per pull, overwriting the random truths. A fresh write head (batch 9) or a newcomer does not hold the convention, so its truths persist. The registered prepared readings did not cover this outcome; it strengthens "mortality keeps honesty current" against the reviewer's alternative.
- **The lock is not absolute.** A strong credit (+0.5) moves immortal liars, slowly (0.65–0.76 after six lifetimes, still rising) where mortal populations switch within one lifetime. Near the threshold, where incentives are small, the lock holds; turnover is what lets weak incentives act. Asymmetry: honest immortals erode under negative credit because readers follow honest entries, so writers feel the penalty; liars are shielded because readers avoid their entries, except for the few who follow, through whom a strong credit leaks.
- **Silence does not slow renewal.** The Discussion's tension is resolved the other way: agents silent at the switch never trained their write heads under the old incentive, so when they start writing they carry no lie. Silence costs a shared wiki dearly instead (population reward 0.65 vs 0.81 at 150 updates), consistent with the earlier finding that silence is unnecessary where the living maintain the record.
- **Honesty vs information.** A penalty for any reader's win is shared by every author the reader saw and does not depend on what an author wrote, so an author's own entry barely changes what it pays (about −0.5/15 ≈ −0.03 per win). It acts like a weak named rivalry (hidden: 28% honest, as at c = −0.02) and does not destroy information. Honesty follows each author's marginal incentive, not the collective harm.
- **Timescale rule.** The forward test failed. The value of an inheritance is measured over a newborn's first 50 updates, so it saturates once protection covers that window, whatever the reading time (shares at 10/25/50 updates: 0.21/0.55/0.96 default, 0.13/0.46/1.03 at half the learning rate, 0.34/0.70/1.04 at double). Absolute values scale with reading speed (+0.053, +0.178, +0.306 at 50 updates). The post hoc "half value at protection ≈ reading time" rule should be dropped; keep: protection must cover the period in which the heir learns to read, and faster readers gain more.
- **Fewer copies.** Both headline results hold with 64 copies.
- **Registered tally after batch 10:** 64 predictions; 37 held, 9 in part, 18 failed.
