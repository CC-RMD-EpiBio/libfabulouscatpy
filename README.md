# libfabulouscatpy

A computer adaptive testing library and simulation framework written in Python.

## Installation

```bash
pip install -e .
```

## Simulating the RWA Scale

The Right-Wing Authoritarianism (RWA) scale is a 22-item assessment with two
factors (scales A and B) calibrated under a multidimensional Graded Response
Model. The item calibrations are shared with
[gofluttercat](https://github.com/CC-RMD-EpiBio/gofluttercat) and live in
`backend-golang/rwas/factorized/`.

### Loading items from gofluttercat

```python
import json
from pathlib import Path

import numpy as np

from libfabulouscatpy.irt.prediction.grm import MultivariateGRM
from libfabulouscatpy.irt.scoring.bayesian import BayesianScoring
from libfabulouscatpy.cat.itemselectors.crossentropy import CrossEntropyItemSelector
from tools.simulation import CATSimulator

# -- 1. Load the item pool from the gofluttercat directory ----------------

ITEMS_DIR = Path("../gofluttercat/backend-golang/rwas/factorized")

items = []
for p in sorted(ITEMS_DIR.glob("*.json")):
    items.append(json.loads(p.read_text()))

# A lightweight stand-in for ItemDatabase / ScaleDatabase
class ItemDB:
    def __init__(self, items):
        self.items = items

class ScaleDB:
    def __init__(self):
        self.scales = {
            "A": {"name": "A", "loc": 0, "scale": 1},
            "B": {"name": "B", "loc": 0, "scale": 1},
        }

# -- 2. Build the IRT model ----------------------------------------------

model = MultivariateGRM(
    itemdb=ItemDB(items),
    scaledb=ScaleDB(),
    interpolation_pts=np.arange(-6.0, 6.0, step=0.05),
)
```

### Running a single adaptive session

```python
# -- 3. Configure the simulator -------------------------------------------

scales = ScaleDB().scales

simulator = CATSimulator(
    model=model,
    selector_class=CrossEntropyItemSelector,
    selector_kwargs={
        "items": items,
        "scales": scales,
        "model": model,
        "temperature": 0.01,
        "max_responses": 12,      # stop after 12 items per scale
        "min_responses": 4,       # at least 4 before early stopping
        "precision_limit": 0.33,  # SE threshold for early stopping
    },
    max_items=22,
    seed=42,
)

# True ability: moderate on A, low on B
theta = {"A": np.array([0.5]), "B": np.array([-1.0])}

result = simulator.run_single(theta)

print(f"Items administered: {result.n_items}")
for step in result.steps:
    scores = {s: f"{sc.score:.2f} (SE {sc.error:.2f})"
              for s, sc in step.scores.items()}
    print(f"  {step.item_id:>4s}  resp={step.response}  {scores}")
```

### Monte Carlo simulation across replicates

```python
# -- 4. Run many replicates to evaluate CAT performance -------------------

summary = simulator.simulate(theta, n_replicates=100)

for scale in summary.scales:
    final_l2 = summary.mean_l2[scale][-1]
    final_se = summary.mean_se[scale][-1]
    print(f"Scale {scale}: final L2 = {final_l2:.3f}, final SE = {final_se:.3f}")
```

### Comparing item selection strategies

```python
from libfabulouscatpy.cat.itemselectors.bayesianfisher import BayesianFisherItemSelector
from libfabulouscatpy.cat.itemselectors.kl import KLItemSelector

for name, cls in [
    ("Cross-Entropy", CrossEntropyItemSelector),
    ("Bayesian Fisher", BayesianFisherItemSelector),
    ("KL Divergence", KLItemSelector),
]:
    sim = CATSimulator(
        model=model,
        selector_class=cls,
        selector_kwargs={
            "items": items,
            "scales": scales,
            "model": model,
            "temperature": 0.01,
            "max_responses": 12,
            "min_responses": 4,
            "precision_limit": 0.33,
        },
        max_items=22,
        seed=42,
    )
    s = sim.simulate(theta, n_replicates=50)
    for scale in s.scales:
        print(f"  {name:20s}  Scale {scale}: "
              f"L2={s.mean_l2[scale][-1]:.3f}  SE={s.mean_se[scale][-1]:.3f}")
```

### Using the embedded item pool

If `libfabulouscatpy` is installed with the RWA items bundled, you can load them
directly without pointing at the gofluttercat directory:

```python
from libfabulouscatpy.rwas.loading import ItemDatabase, ScaleDatabase

model = MultivariateGRM(
    itemdb=ItemDatabase(),
    scaledb=ScaleDatabase(),
)
```

## Item Selection Strategies

| Selector | Description |
|---|---|
| `CrossEntropyItemSelector` | Minimizes expected posterior cross-entropy |
| `BayesianFisherItemSelector` | Maximizes expected Bayesian Fisher information |
| `KLItemSelector` | Maximizes expected KL divergence between prior and posterior |
| `FisherItemSelector` | Maximizes classical Fisher information at the point estimate |
| `GlobalInfoSelector` | Maximizes global information |
| `VarianceItemSelector` | Minimizes expected posterior variance |

Each selector supports stochastic variants (temperature-controlled softmax
sampling) and hybrid modes for balancing exploration vs. exploitation.

## Project Structure

```
libfabulouscatpy/
  cat/                   # CAT engine
    itemselection.py     # Base ItemSelector class
    itemselectors/       # Concrete selection strategies
    session.py           # Session state tracking
  irt/                   # Item Response Theory
    prediction/
      grm.py             # Graded Response Model (multidimensional)
    scoring/
      bayesian.py        # Bayesian posterior scoring
  rwas/                  # Embedded RWA item calibrations
    factorized/          # 22 items, 2-factor model
    autoencoded/         # 22 items, neural network variant
    loading.py           # ItemDatabase / ScaleDatabase loaders
tools/
  simulation.py          # CATSimulator framework
tests/
  tools/
    test_simulation.py   # Simulation tests
```
