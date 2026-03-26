#!/usr/bin/env python3
"""Train MICE LOO imputation models for each dataset and export as JSON.

Requires bayesianquilts (run from the bayesianquilts venv or install it).

Usage:
    cd ~/workspace/bayesianquilts && uv run python \
        ~/workspace/libfabulouscatpy/examples/train_imputation_models.py \
        --datasets grit tma npi eqsq
"""

import argparse
import gc
import json
import os
import sys

import numpy as np

DATASET_CONFIGS = {
    'grit': {'module': 'bayesianquilts.data.grit', 'n_top_features': 12},
    'rwa': {'module': 'bayesianquilts.data.rwa', 'n_top_features': 22},
    'npi': {'module': 'bayesianquilts.data.npi', 'n_top_features': 40},
    'tma': {'module': 'bayesianquilts.data.tma', 'n_top_features': 14},
    'wpi': {'module': 'bayesianquilts.data.wpi', 'n_top_features': 20},
    'eqsq': {'module': 'bayesianquilts.data.eqsq', 'n_top_features': 30},
}

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def mice_to_pairwise_json(mice_loo, item_keys, n_categories):
    """Convert a MICEBayesianLOO model to PairwiseImputationModel JSON format.

    For each pair (target, predictor), we query the MICE model with
    items={predictor: k} for each possible response k to get the
    conditional PMF P(target | predictor=k).  This produces the same
    pairwise PMF table that ``PairwiseImputationModel.from_json()``
    expects.
    """
    pairwise_pmfs = {}
    stacking_weights = {}

    for target in item_keys:
        pairwise_pmfs[target] = {}
        stacking_weights[target] = {}

        for predictor in item_keys:
            if predictor == target:
                continue

            pairwise_pmfs[target][predictor] = {}
            stacking_weights[target][predictor] = 1.0

            for resp_val in range(n_categories):
                try:
                    pmf = mice_loo.predict_pmf(
                        items={predictor: float(resp_val)},
                        target=target,
                        n_categories=n_categories,
                    )
                    pmf = np.asarray(pmf, dtype=float)
                    pmf = np.maximum(pmf, 1e-20)
                    pmf /= pmf.sum()
                    pairwise_pmfs[target][predictor][resp_val] = pmf.tolist()
                except Exception:
                    pairwise_pmfs[target][predictor][resp_val] = \
                        [1.0 / n_categories] * n_categories

    return {
        'n_categories': n_categories,
        'pairwise_pmfs': pairwise_pmfs,
        'stacking_weights': stacking_weights,
    }


def train_dataset(dataset_name, output_dir):
    """Train MICE LOO and export as JSON for one dataset."""
    import importlib
    from bayesianquilts.imputation.mice_loo import MICEBayesianLOO

    config = DATASET_CONFIGS[dataset_name]
    mod = importlib.import_module(config['module'])
    item_keys = mod.item_keys
    response_cardinality = mod.response_cardinality

    print(f"\n{'='*60}")
    print(f"Training MICE LOO for: {dataset_name.upper()}")
    print(f"  Items: {len(item_keys)}, K: {response_cardinality}")
    print(f"{'='*60}")

    # Load data
    df, num_people = mod.get_data(polars_out=True)
    pandas_df = df.select(item_keys).to_pandas()
    pandas_df = pandas_df.replace(-1, np.nan)
    print(f"  People: {num_people}")
    print(f"  Missing per item: min={pandas_df.isna().sum().min()}, "
          f"max={pandas_df.isna().sum().max()}")

    # Fit MICE LOO
    mice_loo = MICEBayesianLOO(
        random_state=42,
        prior_scale=1.0,
        pathfinder_num_samples=100,
        pathfinder_maxiter=50,
        batch_size=512,
        verbose=True,
    )
    mice_loo.fit_loo_models(
        pandas_df,
        n_top_features=config['n_top_features'],
        n_jobs=1,
        fit_zero_predictors=True,
        seed=42,
    )

    # Save the YAML to bayesianquilts notebooks for reference
    bq_dir = os.path.expanduser(
        f'~/workspace/bayesianquilts/notebooks/irt/{dataset_name}')
    if os.path.isdir(bq_dir):
        yaml_path = os.path.join(bq_dir, 'mice_loo_model.yaml')
        mice_loo.save(yaml_path)
        print(f"  Saved YAML: {yaml_path}")

    # Export as JSON for libfabulouscatpy
    os.makedirs(output_dir, exist_ok=True)
    json_data = mice_to_pairwise_json(mice_loo, item_keys, response_cardinality)
    json_path = os.path.join(output_dir, 'imputation_model.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    print(f"  Saved JSON: {json_path}")

    del mice_loo, pandas_df, df
    gc.collect()
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description='Train MICE LOO imputation models')
    parser.add_argument('--datasets', nargs='+',
                        default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()))
    args = parser.parse_args()

    for ds in args.datasets:
        output_dir = os.path.join(EXAMPLE_DIR, ds)
        try:
            train_dataset(ds, output_dir)
        except Exception as e:
            print(f"ERROR training {ds}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
