# WikiBandits: code and data for "How Mortal Learners Keep a Shared Record Useful and Honest"

This repository holds the simulator, the run notebooks, every result table, the scripts that draw the figures, and the dated log of registered predictions behind the paper *How Mortal Learners Keep a Shared Record Useful and Honest* (submitted to *Artificial Life*).

WikiBandits is a simulated world in which sixteen learning agents play slot machines ("bandits"), die and are replaced, and share a written record: each agent owns a page with one entry per machine, and every agent reads all pages while it plays. The paper asks when such a record stays useful to newcomers and when it stays honest.

## Contents

| Path | What it is |
|---|---|
| `wikibandits.py` | The simulator and the learners (JAX; independent PPO with shared weights across copies of the world). All world rules, deaths, credit, juvenile periods, selection and measures are defined here. |
| `baselines.py` | Reference strategies of main-text Table 2 (random play, best player without memory by dynamic programming, Bayesian players, omniscient player). |
| `analysis_wb.py` | Helpers for the measures reported in the paper (time courses, recovery times, dips). |
| `WikiBandits.ipynb` | The run notebook. Sections 13 to 25 are the paper runs, in the order they were made (batches 3 to 12); earlier sections are exploratory and switched off. Sections 24 and 25 hold the corrections made after an independent review (group selection with staggered deaths, credit only for entries a reader can see, and group selection at every death). It embeds `wikibandits.py`, so it runs on its own. |
| `WikiBandits_quick.ipynb` | The same notebook with tiny settings, for a check on a CPU in a few minutes. |
| `GPU_on_Windows.ipynb` | Sets up a GPU JAX environment under WSL2 on Windows and opens the run notebook. |
| `notebooks_as_run/` | The notebook exactly as run for batches 9 to 12 (outputs removed). |
| `build_notebook.py`, `build_launcher.py` | Generate the notebooks above from `wikibandits.py` and the cell texts. |
| `test_wikibandits.py`, `test_launcher.py` | Unit tests of the world rules and the notebook builder (`pytest`). |
| `journal_data/` | Every result table used in the paper (see below). |
| `journal_figs.py`, `journal_fig5.py`, `make_fig1.py` | Draw the paper's figures from `journal_data/` (`make_fig1.py` needs `rsvg-convert`). `journal_fig_survival.py` draws a figure not used in the paper. |
| `recover_fig15.py` | Recovers time courses from the vector paths of the notebook's own figure PDFs (used for some smoothed curves; see *Provenance*). |
| `analysis/` | Post hoc analyses: the timescale analysis, the cross-validated freshness model, the bootstrap interval of the best juvenile period. |
| `registration/` | The registered predictions, copied verbatim from the dated planning log (see *Registration*). |

## Result tables (`journal_data/`)

- `README.md`: which tables are final, the measurement windows, and the older verdict labels kept in early tables.
- `table_S1_all_settings.csv`: Supplementary Table S1, one row per setting with every measure and its standard error; `settings_key.csv` translates each setting code into its changes from the default world and run.
- `summary.csv`: the same measures as used by the figure scripts; `summary_batch8.csv` to `summary_batch11.csv` are the tables as they stood after those batches.
- `contrasts.csv`, `contrasts2.csv`, `contrasts3.csv`: registered contrasts with 95% intervals and verdicts.
- `honesty_threshold.csv`, `juvenile_dose.csv`, `evolution.csv`, `events.csv`, `timescale.csv`, `adult_history.csv`, `adult_verdicts.csv`, `extra2.csv`, `extra3.csv`: the tables behind specific results (threshold, dose, selection, die-off and sleeper events, timescale, adult-only history).
- `batch7/` to `batch12/`: the tables and figures each later batch of runs produced, including the verdicts of its registered predictions.
- `*.npz`: smoothed time courses for the figures (load with `numpy.load(..., allow_pickle=False)`).

## Reproducing

**Figures from the result tables** (minutes, CPU):

```
pip install -r requirements.txt
python journal_figs.py journal_data figures_out
python journal_fig5.py journal_data figures_out
python make_fig1.py figures_out
```

Main-text Figures 1 to 5 are `fig1_world`, `fig2_newcomers`, `fig4_inheritance`, `fig6_evolution_long` and `fig5_turnover_v3`; Supplementary Figures S1 to S7 are `fig3_robustness`, `figS_bottleneck`, `fig5_evolution`, `fig8_identity`, `fig9_symbols`, `fig7_honesty` and `fig_turnover` (the file names are historical).

**The runs** (about 24 GPU-hours on one RTX 3070 for all 1,735 runs): open `WikiBandits.ipynb` on a machine with an NVIDIA GPU and JAX with CUDA (or in Google Colab with a GPU runtime) and run all cells. Results are saved after each run, and a re-run resumes where it stopped. The notebook refuses to start the paper runs on a CPU. Each paper section of the notebook writes the tables found in `journal_data/`.

**Tests:** `pytest test_wikibandits.py`.

## Provenance

Most figure panels are drawn directly from the tables in `journal_data/`. Time courses of the credit-switch runs (`history_series.npz`, `batch10/batch10_series.npz`) were saved directly by the notebook. The per-update logs of the evolution runs were not kept, so their smoothed time courses (`evolution_series.npz`, used for main-text Figure 4 and Supplementary Figure S3) were recovered from the vector paths of the notebook's own figure PDFs; their end points match `evolution.csv`. Three older recovered files (`hysteresis_series.npz` and `hysteresis_adult_series.npz`, recovered with `recover_fig15.py`, and `survival_series.npz`) are not used by the current figures.

## Registration

From the third batch of runs on, predictions were written down, dated, before the runs they concern, in a private planning log. `registration/predictions_log.md` copies those sections verbatim; `registration/batch10_predictions.md` is the separate note written before batch 10. The log was not deposited in a public registry, so its dates are as written at the time. Supplementary Table S5 of the paper reports every prediction against its criterion.

## Licence

The code (all `.py` files and notebooks) is released under the MIT licence (`LICENSE`). The result tables, analysis outputs and registration notes are released under CC BY 4.0 (`LICENSE-data.md`).

## Citation

See `CITATION.cff`.
