"""Derive a PairwiseImputationModel analytically from an IRT model.

For each pair of items (i, j), the conditional PMF is computed by
marginalizing over the latent ability θ using the model prior:

    P(x_i=k | x_j=l) = ∫ P(x_i=k|θ) P(x_j=l|θ) π(θ) dθ
                        / ∫ P(x_j=l|θ) π(θ) dθ

This produces a ``PairwiseImputationModel`` that captures the marginal
dependencies between items as implied by the IRT model, without relying
on external calibration data.
"""

import numpy as np

from libfabulouscatpy._compat import trapz as _trapz
from libfabulouscatpy.imputation.pairwise import PairwiseImputationModel


def pairwise_imputation_from_grm(grm, scale_name, interpolation_pts=None,
                                  prior_sigma=1.0):
    """Build a PairwiseImputationModel from a MultivariateGRM.

    Parameters
    ----------
    grm : MultivariateGRM
        Fitted IRT model.
    scale_name : str
        Which scale to derive pairwise PMFs for.
    interpolation_pts : ndarray, optional
        Theta grid for numerical integration.  Defaults to the model's grid.
    prior_sigma : float
        Standard deviation of the Gaussian prior on θ.

    Returns
    -------
    PairwiseImputationModel
    """
    model = grm.models[scale_name]
    item_labels = model.item_labels
    pts = interpolation_pts if interpolation_pts is not None else grm.interpolation_pts

    # P(x_i=k|θ) for all items: shape (n_theta, n_items, K)
    log_p_all = model.log_likelihood(theta=pts, observed_only=False)
    p_all = np.exp(log_p_all)
    n_items, n_categories = p_all.shape[1], p_all.shape[2]

    # Prior: shape (n_theta,)
    prior = np.exp(-0.5 * (pts / prior_sigma) ** 2)
    prior /= _trapz(y=prior, x=pts)

    pairwise_pmfs = {}
    stacking_weights = {}

    for i, target in enumerate(item_labels):
        pairwise_pmfs[target] = {}
        stacking_weights[target] = {}

        for j, predictor in enumerate(item_labels):
            if i == j:
                continue

            pairwise_pmfs[target][predictor] = {}
            stacking_weights[target][predictor] = 1.0

            for l in range(n_categories):
                # P(x_j=l|θ) * π(θ): shape (n_theta,)
                weight = p_all[:, j, l] * prior
                denom = _trapz(y=weight, x=pts)
                if denom < 1e-30:
                    # This response level is essentially impossible
                    pmf = np.ones(n_categories) / n_categories
                else:
                    # P(x_i=k | x_j=l) = ∫ P(x_i=k|θ) P(x_j=l|θ) π(θ) dθ / denom
                    pmf = _trapz(
                        y=p_all[:, i, :] * weight[:, np.newaxis],
                        x=pts,
                        axis=0,
                    ) / denom

                pmf = np.maximum(pmf, 1e-20)
                pmf /= pmf.sum()
                pairwise_pmfs[target][predictor][l] = pmf.tolist()

    return PairwiseImputationModel(
        pairwise_pmfs=pairwise_pmfs,
        stacking_weights=stacking_weights,
        n_categories=n_categories,
    )
