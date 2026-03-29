#!/usr/bin/env python3
"""Generate LaTeX tables from simulation results.

Reads .npz accuracy files and produces tables with mean (SD) format,
method initials, and ability-range stratification.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

TRUE_ABILITIES = np.arange(-3, 3.5, 0.5)
RANGES = {
    'low':  TRUE_ABILITIES <= -1.5,
    'mid':  (TRUE_ABILITIES >= -1.0) & (TRUE_ABILITIES <= 1.0),
    'high': TRUE_ABILITIES >= 1.5,
    'all':  np.ones(len(TRUE_ABILITIES), dtype=bool),
}

# Method key -> file suffix, display initial
METHODS = [
    ('fisher',             'F'),
    ('bayesian_fisher',    'BF'),
    ('global_info',        'GI'),
    ('bayesian_variance',  'BV'),
    ('entropy',            'E'),
    ('stochastic_entropy', 'SE'),
]

# Also try old names for backward compat
OLD_NAMES = {
    'entropy': 'cross_entropy',
    'stochastic_entropy': 'stochastic_ce',
}

DATASETS = {
    'grit': {'label': 'GRIT', 'items': 12, 'K': 5,  'test_lengths': [5, 10]},
    'npi':  {'label': 'NPI',  'items': 40, 'K': 2,  'test_lengths': [5, 10, 20]},
    'tma':  {'label': 'TMA',  'items': 50, 'K': 2,  'test_lengths': [5, 10]},
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_metric(dataset, method_key, metric):
    """Load a metric array from the .npz file."""
    results_dir = os.path.join(SCRIPT_DIR, dataset, 'results')
    path = os.path.join(results_dir, f'{dataset}_accuracy_{method_key}.npz')
    if not os.path.exists(path):
        alt = OLD_NAMES.get(method_key)
        if alt:
            path = os.path.join(results_dir, f'{dataset}_accuracy_{alt}.npz')
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data[metric]


def fmt(mean, sd):
    """Format as mean (SD) with 2 decimal places."""
    return f'{mean:.2f} ({sd:.2f})'


def generate_l2_table():
    """Generate the L2 error table."""
    method_initials = ' & '.join(m[1] for m in METHODS)
    n_methods = len(METHODS)

    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{\textbf{Absolute score error} '
          r'$|\hat\theta_{\text{CAT}} - \hat\theta_{\text{full}}|$, '
          r'mean (SD), under $\mathcal{M}$-open simulation (500 replicates per ability).')
    print(r'Lower is better; \textbf{bold} = best mean per row.')
    print(r'F = Fisher; BF = Bayesian Fisher; GI = global information; '
          r'BV = Bayesian variance; E = entropy (greedy); SE = entropy (stochastic).}')
    print(r'\label{tab:mopen_l2}')
    print(r'\small')
    cols = '@{}ll' + ' r' * n_methods + '@{}'
    print(r'\begin{tabular}{' + cols + '}')
    print(r'\toprule')
    print(r'Dataset & $\theta$ & ' + method_initials + r' \\')
    print(r'\midrule')

    for ds_key, ds_info in DATASETS.items():
        for ti, tl in enumerate(ds_info['test_lengths']):
            if ti == 0:
                print(rf'\multicolumn{{{n_methods + 2}}}{{@{{}}l}}'
                      rf'{{\textit{{{ds_info["label"]} '
                      rf'({ds_info["items"]} items, $K{{=}}{ds_info["K"]}$), '
                      rf'$t = {tl}$}}}} \\')
            else:
                print(rf'\multicolumn{{{n_methods + 2}}}{{@{{}}l}}'
                      rf'{{\textit{{$t = {tl}$}}}} \\')

            step_idx = tl - 1
            for rname, mask in RANGES.items():
                vals = []
                for mkey, _ in METHODS:
                    arr = load_metric(ds_key, mkey, 'l2')
                    if arr is None or step_idx >= arr.shape[2]:
                        vals.append((np.nan, np.nan))
                        continue
                    subset = arr[mask, :, step_idx]
                    vals.append((np.nanmean(subset), np.nanstd(np.nanmean(subset, axis=1))))

                means = [v[0] for v in vals]
                best_val = np.nanmin(means)
                cells = []
                for i, (m, s) in enumerate(vals):
                    cell = fmt(m, s)
                    if round(m, 2) <= round(best_val, 2):
                        cell = r'\textbf{' + cell + '}'
                    cells.append(cell)

                print(f'  & {rname} & ' + ' & '.join(cells) + r' \\')

            if ds_key != list(DATASETS.keys())[-1] or tl != ds_info['test_lengths'][-1]:
                print(r'\midrule')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


def generate_kl_table():
    """Generate the KL divergence table (overall only)."""
    method_initials = ' & '.join(m[1] for m in METHODS)
    n_methods = len(METHODS)

    print(r'\begin{table}[t]')
    print(r'\centering')
    print(r'\caption{\textbf{KL divergence} '
          r'$\mathcal{D}(\pi(\theta|\bx) \| \tpi(\theta|\bx_t))$, '
          r'mean (SD), under $\mathcal{M}$-open simulation.')
    print(r'Lower is better; \textbf{bold} = best per row. '
          r'$\uparrow$ = increases with $t$ (pathological).')
    print(r'F = Fisher; BF = Bayesian Fisher; GI = global information; '
          r'BV = Bayesian variance; E = entropy (greedy); SE = entropy (stochastic).}')
    print(r'\label{tab:mopen_kl}')
    print(r'\small')
    cols = '@{}ll' + ' r' * n_methods + '@{}'
    print(r'\begin{tabular}{' + cols + '}')
    print(r'\toprule')
    print(r'Dataset & $t$ & ' + method_initials + r' \\')
    print(r'\midrule')

    prev_kl = {}  # track previous KL for pathological detection
    for ds_key, ds_info in DATASETS.items():
        for tl in ds_info['test_lengths']:
            step_idx = tl - 1
            mask = RANGES['all']
            vals = []
            for mkey, minitial in METHODS:
                arr = load_metric(ds_key, mkey, 'kl')
                if arr is None or step_idx >= arr.shape[2]:
                    vals.append((np.nan, np.nan))
                    continue
                subset = arr[mask, :, step_idx]
                vals.append((np.nanmean(subset), np.nanstd(np.nanmean(subset, axis=1))))

            means = [v[0] for v in vals]
            best_val = np.nanmin(means)
            cells = []
            for i, (m, s) in enumerate(vals):
                cell = fmt(m, s)
                if round(m, 2) <= round(best_val, 2):
                    cell = r'\textbf{' + cell + '}'
                # Check if KL increased from previous test length
                prev = prev_kl.get((ds_key, METHODS[i][0]))
                if prev is not None and m > prev + 0.01:
                    cell += r'$\,\uparrow$'
                cells.append(cell)
                prev_kl[(ds_key, METHODS[i][0])] = m

            print(f'  {ds_info["label"]} & {tl} & ' + ' & '.join(cells) + r' \\')

        if ds_key != list(DATASETS.keys())[-1]:
            print(r'\midrule')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')


if __name__ == '__main__':
    print('% === L2 TABLE ===')
    generate_l2_table()
    print()
    print('% === KL TABLE ===')
    generate_kl_table()
