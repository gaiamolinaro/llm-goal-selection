# Data Analysis Code for "Language Model Goal Selection Differs from Humans' in a Self-Directed Learning Task"

This repository contains the full data analysis code accompanying the NeurIPS 2026 paper *Language Model Goal Selection Differs from Humans' in a Self-Directed Learning Task* (Molinaro, August, Perszyk, & Collins). To avoid data contamination in future models, we do not publish any raw data. 

## Setup

```bash
pip install -r requirements.txt
```

Edit the paths at the top of [settings.py](settings.py) and
[holton/settings.py](holton/settings.py) to point at your own data
directories before running any notebook. Notebooks also save figures to a
relative `results/figures/` folder — create it (or change `plots_dir` in the
notebook) before running. `.gitignore` in this repo excludes `data/`,
`results/`, and common data file extensions, so your own data and outputs
won't accidentally get committed here.

## Repository structure

```
root/
├── data_analysis.ipynb
├── data_analysis_cross_exp.ipynb
├── helpers.py
├── settings.py
└── holton/
    ├── goal_commitment.ipynb
    ├── helpers.py
    ├── settings.py
    └── restructure_all_participants.py
```

- **`data_analysis.ipynb`** — the full single-experiment analysis pipeline: load
  processed data, compute metrics, and produce performance/goal-selection
  figures and comparison tables for one experiment variant at a time.
- **`data_analysis_cross_exp.ipynb`** — compares several experiment variants
  against each other and against humans on the same figures/tables.
- **`helpers.py`** — shared plotting (`Plotter` class) and statistics helper
  functions used by both notebooks above, trimmed to just what they call (the
  original project's helper module also supported other analyses not part of
  this paper).
- **`settings.py`** — data directory configuration (edit before running).
- **`holton/goal_commitment.ipynb`** — analysis for the separate
  goal-commitment ("fishing net") replication (paper Appendix "Comparing Goal
  Commitment in Humans and LLMs").
- **`holton/helpers.py`** / **`holton/settings.py`** — helper functions and
  data directory configuration specific to that task.
- **`holton/restructure_all_participants.py`** — converts Holton et al.'s
  original published human data into the schema `goal_commitment.ipynb`
  expects (see [Data format](#data-format) below). Reads/writes
  `all_participants.csv`/`all_participants_restructured.csv` under the
  `data_dir` configured in `holton/settings.py`; their raw data itself is
  not included.

`data_analysis.ipynb` selects one experiment variant near the top via a
`version_dir` string (e.g. `""`, `"V2/"`, ... `"V8/"`), which is appended to
`data_dir`/`out_dir` to point at a differently-processed data snapshot.
`data_analysis_cross_exp.ipynb` instead takes a `version_dirs` list so you can
compare several variants at once. In our own project these variants
corresponded to the persona-steering, skill-acquisition/non-words semantics,
interactive (computer-using-agent), and context-window manipulations
described in the paper; adjust the branches/dictionaries in each notebook's
config cell to whatever directory layout you use for your own data versions.
A few cells also derive the skill-acquisition/non-words variants' data
directory by string-replacing a task-name substring in `data_dir` (e.g.
`.replace("Alchemist", "Acrobat")`) — this assumes our own raw-data naming
convention and will need to be adapted to yours.

## Data format

To reuse the analysis code, please use input data structured as detailed below.

### Main task (alchemy / skill-acquisition / non-words variants)

`data_analysis.ipynb` and `data_analysis_cross_exp.ipynb` expect processed dataframes at
`out_dir/<family>/<model>/{task_df,train_df,learn_df,test_df}.csv`, where
`family` is one of `openai`, `google`, `anthropic`, `qwen`, `centaur`
(see `helpers.get_family`) — plus the equivalent
`data/human/*.csv` set for human data. 

`task_df` / `train_df` / `learn_df` / `test_df` (one row per event within a
trial — goal selection, each action selection, and feedback):

| Column | Description |
|---|---|
| `exp_id` | Experiment/task version identifier |
| `source` | Data source: model name (LLM data) or recruitment platform (human data) |
| `subj_id` | Anonymized subject/simulation identifier |
| `subj_num` | Integer subject number, assigned in participation order |
| `block` | Block number within the task |
| `trial` | Trial number (1-indexed) |
| `trial_in_block` | Trial number within the current block |
| `condition` | Task configuration id (one of 10 randomized potion/ingredient/color assignments) |
| `phase` | Task phase: 0 = practice/train, 1 = learning, 2 = test |
| `event_id` | Event type label: `goal_selection`, `action_selection`, or `feedback` |
| `choice` | Selected item's on-screen/list position |
| `choice_id` | Selected item's underlying identity (potion/ingredient id) |
| `correct` | Whether the trial's final ingredient sequence matched the goal's recipe (1/0), on feedback rows |
| `feedback` | Feedback shown to the participant (same coding as `correct`) |
| `chosen_goal`, `chosen_goal_position` | The goal id / on-screen position selected on this trial (repeated across all of the trial's rows) |
| `temperature` | LLM data only: sampling setting used for that call |

### Goal-commitment task (`holton/`)

`holton/goal_commitment.ipynb` expects `data_dir/<family>/<model>/learn_df.csv`
(one row per round of the fishing-net task):

| Column | Description |
|---|---|
| `participant` | Anonymized subject/simulation identifier |
| `block_id`, `trial_in_block` | Round numbering (a new block starts each time a net is filled) |
| `net_type` | Which seafood type (crab/octopus/fish) currently fills the net |
| `net_amount_before`, `net_size` | Net contents before the round, and the net's capacity |
| `offer_crab`, `offer_octopus`, `offer_fish` | Offered quantities for each seafood type this round |
| `choice`, `choice_offer` | Chosen seafood type and its offered amount |
| `switched` | Whether the choice differed from `net_type` |
| `net_filled`, `points_scored` | Whether the net was filled this round, and cumulative points |

For the human comparison data, this table is produced from Holton et al.'s data (obtained from the published OSF repository,
<https://osf.io/mvquk/>) by `holton/restructure_all_participants.py`.

## Citation

If you use this code, please cite the paper it accompanies:

> Molinaro, G., August, D., Perszyk, D., & Collins, A. G. E. (2026). Language
> Model Goal Selection Differs from Humans' in a Self-Directed Learning Task.
> Advances in Neural Information Processing Systems, 39.

```bibtex
@inproceedings{molinaro2026goalselection,
  title     = {Language Model Goal Selection Differs from Humans' in a Self-Directed Learning Task},
  author    = {Molinaro, Gaia and August, Dave and Perszyk, Danielle and Collins, Anne G. E.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026}
}
```
