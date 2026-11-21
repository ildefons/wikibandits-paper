# Result tables

**Start here.** `table_S1_all_settings.csv` (Supplementary Table S1) has one row per setting with every measure, its standard error and the number of seeds. `settings_key.csv` translates each setting code into its changes from the default world (`wikibandits.BanditConfig`) and from the default run (300-update lives, six lifetimes, 256 copies of the world, the default learning rate).

**Measurement windows.** Newborn reward (mean reward per pull over the first 50 updates of a life) and adult reward (last 50 updates of a complete life) average over every life after the founders', across the whole run. Population reward, honest writes, wiki right, writes and follows average over the last third of the run. Recovery time ("t to 0.61") is the median, over complete lives, of the number of updates until a newborn's reward, smoothed over 5 updates, first reaches 0.61. Honest writes is the share of an agent's writes that report its belief, averaged over agents and updates. In the credit-switch runs (`hyst_*`), newborn and adult reward therefore mix lives before and after the switch; `batch11/batch11_post_switch.csv` gives them over lives born at least one lifetime after the switch.

**Which tables are final.** The verdict rule changed after batch 7. The paper uses the rule of `batch8/stats_verdicts.csv` and later files: *equivalent* (the 90% interval lies inside ±m and the 95% interval includes zero), *small but non-zero*, *larger than m*, *different* and *inconclusive*. Older files keep the labels in use when they were written:

| File | Status |
|---|---|
| `contrasts.csv`, `contrasts2.csv`, `contrasts3.csv` | Earlier rounds; labels "same within ±0.02", "different" and "unclear" from the old rule, and the seeds available then. Superseded where `batch8/stats_verdicts.csv` repeats a comparison with more seeds. |
| `juvenile_dose.csv` | Three-seed values, superseded by the ten-seed values in `summary.csv` (for example +0.178 rather than +0.18 for a 50-update juvenile period). |
| `summary_batch8.csv`, `summary_batch9.csv`, `summary_batch10.csv`, `summary_batch11.csv` | Snapshots of `summary.csv` after those batches. |
| `evo_wiki_group_from0_long`, `evo_wiki_group_from150_long` (rows of `summary.csv`) | Superseded by `evo_wiki_groupb_*`. The original group selection replaced the juvenile periods of living agents at each selection (Supplementary Section S4); the rows are kept for the record. |
| `notebook*_cited*` rows | Credit was paid for entries no reader could see. The rows ending `_seen` rerun six of these comparisons with credit only for seen entries (`batch11/batch11_seen.csv`, Supplementary Section S8). |
| `batch7/` | Batch 7 as run (three seeds); its verdicts were re-judged with ten seeds in `batch8/`. |
| `batch8/`, `batch9/`, `batch10/`, `batch11/`, `batch12/` | Final for the comparisons they contain. |

Small differences in the third decimal between an interval in the paper and in an older table come from the seeds available when each table was written; the paper uses the final seeds.
