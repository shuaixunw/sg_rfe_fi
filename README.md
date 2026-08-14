# GRACE

Reproducibility archive for the GRACE feature-selection project. It contains
the implementation and generated results supporting the manuscript's five experiments, while deliberately
excluding raw and processed research data.

## Repository layout

```text
GRACE/
├── code/
│   ├── experiment1/   # Euclidean/tabular benchmark and follow-up analyses
│   ├── experiment2/   # Modes A and B geometry experiments (Manuscript Experiments II–III)
│   ├── experiment3/   # Robustness analyses (Manuscript Experiment V)
│   └── experiment4/   # Mode C validation and Random-Forest benchmark
├── data/              # Data acquisition and placement instructions only
├── results/           # Tables, figures, reports, and reproducible outputs
└── tests/             # Shared Experiment I tests
```

The source data are intentionally not tracked. See [data/README.md](data/README.md)
for the dataset catalog, licenses/access conditions, download scripts, and expected
local paths.

## Manuscript-to-code map

| Manuscript experiment | Reproduction entry point |
|---|---|
| Experiment I — Classical Reconstruction | `code/experiment1/` |
| Experiment II — Mode A: Representation-Level Generation | `code/experiment2/` (Mode A configurations) |
| Experiment III — Mode B: Topology-Level Generation | `code/experiment2/` (Mode B configurations) |
| Experiment IV — Mode C: Topology-Semantic Cascade | `code/experiment4/experiment4_run_all.py` |
| Experiment V — Robustness and Sensitivity Analysis | `code/experiment3/` |

Experiments II and III are separate manuscript questions but share the same
geometry-experiment implementation and datasets. Experiment V uses the existing
`experiment3` robustness implementation; there is no fictitious fifth code tree.

## Reproduction package

- `dataset_registry.csv`: dataset identities and verified status fields.
- `requirements.txt`: dependency specification.
- `results/`: derived tables, figures, reports, and failure logs.
- Experiment-specific configuration and seed information in the corresponding
  code and result directories.
- Exact entry points documented here and in experiment-level README files.

## Reproducing Experiment IV

Experiment IV is self-contained after the tabular corpus has been placed at
`GRACE_Corpus/standard_tabular_benchmark/` beside this repository. From the
parent project directory:

```bash
PYTHONPATH=GRACE/code/experiment4:GRACE/code/experiment1 \
python GRACE/code/experiment4/experiment4_run_all.py
```

The archived outputs in `results/experiment4/` were produced with seed 42,
five stratified folds where possible, four feature budgets (5%, 10%, 20%, 50%),
and a 100-tree Random Forest. The complete report is
`results/experiment4/EXPERIMENT4_REPORT.md`.

The sole downstream failure was `pmlb::poker`. Class 9 had only one observation
in the capped test sample, so stratified sampling could not form the required
groups. The failure is retained in `results/experiment4/experiment4_failures.csv`;
no imputation or replacement dataset was used.

## Important notes

- Results are included; raw data, processed feature arrays, download archives,
  caches, and checkpoints are not.
- Some source datasets require registration or acceptance of a data-use agreement.
- The parent project contains the external data directories during local execution;
  their absence from this repository is intentional.
