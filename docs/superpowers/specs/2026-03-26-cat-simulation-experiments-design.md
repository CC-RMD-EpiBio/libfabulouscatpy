# CAT Simulation Experiments on Public Psychometric Instruments

## Goal

Run CAT simulations comparing 6 item selection methods on 4 publicly available psychometric instruments (EQSQ, WPI, TMA, NPI) to demonstrate that the stochastic cross-entropy selector generalizes beyond the WD-FAB. Results feed into the NeurIPS 2026 version of the paper (`cat_optimalcontrol/cat_neurips.tex`).

## Instruments

| Instrument | Items | Response K | Fitted Model | Notes |
|------------|-------|-----------|--------------|-------|
| EQSQ       | 120   | 4         | Yes (grm_baseline) | `bayesianquilts/notebooks/irt/synthetic/results/eqsq/grm_baseline/` |
| WPI        | 116   | 2         | **Needs fitting** | `bayesianquilts/notebooks/irt/wpi/` |
| TMA        | 50    | 2         | **Needs fitting** | `bayesianquilts/notebooks/irt/tma/` |
| NPI        | 40    | 2         | Neural only | Need baseline GRM at `bayesianquilts/notebooks/irt/npi/` |

## Selection Methods (6 total)

1. Fisher Information (greedy, Eq. 3 in paper)
2. Bayesian Fisher Information (Eq. 4)
3. Global Information (Eq. 5)
4. Bayesian Variance (Eq. 6)
5. Cross-Entropy / Discrepancy (greedy, Eq. 8)
6. Stochastic Cross-Entropy (Eq. 9)

## Architecture

Three separate scripts in `libfabulouscatpy/examples/`, run sequentially. Each is memory-independent of the others.

### Script 1: `extract_grm_params.py`

Extracts GRM parameters from fitted bayesianquilts models into lightweight `.npz` files.

**Memory strategy**: Load one model at a time, sample from surrogate (n=1000), compute mean of transformed samples, save `.npz`, delete model + `gc.collect()`.

**Input**: Path to a `grm_baseline/` directory containing `params.h5`.

**Output**: `{instrument}_grm_params.npz` containing:
- `slope`: array of shape `(n_items,)` — discrimination parameters (mean of softplus-transformed surrogate samples)
- `calibration`: array of shape `(n_items, K-1)` — difficulty thresholds (cumsum of difficulties0 + softplus(ddifficulties))
- `item_keys`: list of item names
- `response_cardinality`: int

**Bijector handling**:
- `discriminations`: stored pre-softplus. Sample from surrogate, softplus is applied by the surrogate distribution generator, then take mean.
- `difficulties0`: Identity bijector. Mean of samples = mean of loc.
- `ddifficulties` (K > 2 only): stored pre-softplus. Same as discriminations.
- Calibration reconstruction: `calibration[:, 0] = difficulties0`, `calibration[:, j] = difficulties0 + cumsum(ddifficulties[:, :j])` for j >= 1.

### Script 2: `cat_simulation.py`

Runs CAT simulations using pure numpy (no JAX). Loads `.npz` params and uses libfabulouscatpy's `GradedResponseModel` and item selectors.

**Memory strategy**: One instrument at a time. Each simulation is independent. Results saved incrementally.

**Simulation protocol** (matching the paper):
- True abilities: `theta in {-3, -2.5, -2, ..., 2.5, 3}` (13 values)
- Respondents per ability: 500
- Test lengths recorded: {5, 10, 20, 30, 40} items (or up to n_items if fewer)
- For each respondent: simulate full item responses, then run CAT with each selector

**Metrics collected per (instrument, selector, test_length, true_ability)**:
- KL discrepancy: `D(pi(theta|x) || pi(theta|x_t))`
- Absolute error in mean: `|E[theta|x_t] - E[theta|x]|`
- Posterior SD: `sqrt(Var[theta|x_t])`

**Exposure experiment** (per instrument, selector):
- 12 items per session (or n_items/4, whichever is smaller)
- Abilities drawn from N(0,1)
- Session counts: {8, 16, 32, 64, 128}
- Metric: number of unique items exposed

**Output**: `{instrument}_cat_results.npz` per instrument.

### Script 3: `plot_results.py`

Generates paper figures from `.npz` result files. Produces PDF figures matching the paper's style (faceted by instrument, selector, test length).

**Output figures** (per instrument, saved to `cat_optimalcontrol/figures/`):
- KL discrepancy vs ability by selector and test length
- Absolute error vs ability by selector and test length
- Posterior SD vs ability by selector and test length
- Item exposure vs number of sessions by selector

## Step 0: Fit WPI and TMA

```bash
cd bayesianquilts/notebooks/irt
uv run python run_single_notebook.py --dataset wpi
uv run python run_single_notebook.py --dataset tma
```

Also fit NPI baseline (only neural_grm exists):
```bash
uv run python run_single_notebook.py --dataset npi
```

## Execution Order

```
1. Fit WPI, TMA, NPI baseline GRMs (bayesianquilts)
2. Extract params: uv run python extract_grm_params.py (one per instrument)
3. Run simulations: uv run python cat_simulation.py (one per instrument)
4. Plot: uv run python plot_results.py
```

## Dependencies

- `extract_grm_params.py`: requires bayesianquilts (JAX, TFP)
- `cat_simulation.py`: requires only libfabulouscatpy + numpy (no JAX)
- `plot_results.py`: requires matplotlib + numpy

## Risk: NPI baseline

NPI currently only has a neural_grm fitted (not a standard GRM baseline). Running `run_single_notebook.py --dataset npi` should produce a baseline. If it fails, we fall back to extracting from the neural_grm (different extraction logic but same output format).
