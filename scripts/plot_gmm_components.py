"""
Dev tool: generate 2D visualizations of the fitted FormulaPrior GMM

For each feature pair, produces a side-by-side figure:
    LEFT  : 2D histogram of corpus formulae (the "real" distribution)
    RIGHT : same plot region, filled with overlapping low-alpha ellipses
            from the GMM components — the model's reconstruction of
            that pattern in 2D.

Pairs:
    - H/C vs O/C
    - RDBE/C vs O/C
    - N count vs O/C

Usage:
    uv run python scripts/plot_gmm_components.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from find_mfs.scoring.prior import (
    FormulaPrior,
    _RATIO_FEATURES,
    _COUNT_FEATURES,
    _N_FEATURES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CORPUS_PATH = Path(__file__).resolve().parent.parent / 'COCONUT_molecular_formulae.txt'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'plots'
N_COMPONENTS = 25

# Feature index lookup. Order from prior.py:
#   0: H/C, 1: O/C
#   2..7: N, S, P, Cl, Br, I (counts)
#   8: RDBE, 9: RDBE/C, 10: n_heteroatom_types
FEATURE_NAMES = (
    list(f'{e}/C' for e in _RATIO_FEATURES)
    + list(_COUNT_FEATURES)
    + ['RDBE', 'RDBE/C', 'n_hetero_types']
)
FEATURE_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}

assert len(FEATURE_NAMES) == _N_FEATURES, \
    f'Expected {_N_FEATURES} features, got {len(FEATURE_NAMES)}'

# Pairs to plot: (x_feature, y_feature, x_label, y_label, x_clip, y_clip)
PAIRS = [
    ('H/C',     'O/C', 'H/C ratio',          'O/C ratio',         (0, 3.5),  (0, 1.5)),
    ('RDBE/C',  'O/C', 'RDBE / C ratio',     'O/C ratio',         (0, 1.2),  (0, 1.5)),
    ('N',       'O/C', 'Nitrogen count',      'O/C ratio',         (0, 12),   (0, 1.5)),
]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def marginalize_2d(mean: np.ndarray, cov: np.ndarray, idx_x: int, idx_y: int):
    """
    Marginal mean & covariance of a multivariate Gaussian over two dims
    """
    sub_mean = np.array([mean[idx_x], mean[idx_y]])
    sub_cov = np.array([
        [cov[idx_x, idx_x], cov[idx_x, idx_y]],
        [cov[idx_y, idx_x], cov[idx_y, idx_y]],
    ])
    return sub_mean, sub_cov


def ellipse_from_cov(mean_2d, cov_2d, n_sigma=2.0, **kwargs) -> Ellipse:
    """
    Build a matplotlib Ellipse for the n-sigma contour of a 2D Gaussian
    """
    eigvals, eigvecs = np.linalg.eigh(cov_2d)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    angle_deg = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 2 * n_sigma * np.sqrt(max(eigvals[0], 1e-12))
    height = 2 * n_sigma * np.sqrt(max(eigvals[1], 1e-12))

    return Ellipse(
        xy=mean_2d, width=width, height=height, angle=angle_deg, **kwargs
    )


_FILL_COLOR = 'steelblue'
_FILL_ALPHA = 0.06   # so heavy overlap = darker = denser
_N_SIGMA = 1.0


def plot_pair(
    features: np.ndarray,
    gmm,
    component_weights: np.ndarray,
    x_name: str,
    y_name: str,
    x_label: str,
    y_label: str,
    x_clip: tuple[float, float],
    y_clip: tuple[float, float],
    output_path: Path,
):
    idx_x = FEATURE_IDX[x_name]
    idx_y = FEATURE_IDX[y_name]

    x_data = features[:, idx_x]
    y_data = features[:, idx_y]

    # Mask data to plotting window so the histogram is informative
    mask = (
        (x_data >= x_clip[0]) & (x_data <= x_clip[1])
        & (y_data >= y_clip[0]) & (y_data <= y_clip[1])
    )
    x_plot = x_data[mask]
    y_plot = y_data[mask]

    fig, (ax_data, ax_gmm) = plt.subplots(
        1, 2, figsize=(13, 5.6), sharex=True, sharey=True,
    )
    n_components = gmm.means_.shape[0]

    # ---- Left panel: data ----
    h = ax_data.hist2d(
        x_plot, y_plot,
        bins=120,
        range=[x_clip, y_clip],
        cmap='Blues',
        norm='log',
    )
    cb = fig.colorbar(h[3], ax=ax_data, pad=0.02)
    cb.set_label('formulae per bin (log scale)')
    ax_data.set_title(f'COCONUT data ({len(x_plot):,} formulae)', fontsize=12)

    # ---- Right panel: GMM components, filled, low alpha ----
    ax_gmm.set_facecolor('white')
    for k in range(n_components):
        sub_mean, sub_cov = marginalize_2d(
            gmm.means_[k], gmm.covariances_[k], idx_x, idx_y
        )
        ell = ellipse_from_cov(
            sub_mean, sub_cov,
            n_sigma=_N_SIGMA,
            facecolor=_FILL_COLOR,
            edgecolor='none',
            alpha=_FILL_ALPHA,
        )
        ax_gmm.add_patch(ell)
    ax_gmm.set_title(
        f'GMM ({n_components} components, 2σ fills)', fontsize=12,
    )

    # Shared cosmetics
    for ax in (ax_data, ax_gmm):
        ax.set_xlim(x_clip)
        ax.set_ylim(y_clip)
        ax.set_xlabel(x_label, fontsize=12)
        ax.grid(alpha=0.2)
    ax_data.set_ylabel(y_label, fontsize=12)

    fig.suptitle(
        f'{x_label}  vs  {y_label}     (2D projection of an 11D model)',
        fontsize=13, y=0.99,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f'  wrote {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading corpus from {CORPUS_PATH} ...')
    corpus = CORPUS_PATH.read_text().splitlines()
    print(f'  {len(corpus)} formulae')

    print(f'Fitting GMM with {N_COMPONENTS} components (cache will be used if available) ...')
    prior = FormulaPrior().fit(corpus, n_components=N_COMPONENTS)
    gmm = prior._gmm
    weights = gmm.weights_

    # Feature matrix is cached on disk after the first fit; reuse it.
    features = FormulaPrior._parse_corpus(corpus)
    print(f'  feature matrix: {features.shape}')

    print('Generating plots ...')
    for x_name, y_name, x_label, y_label, x_clip, y_clip in PAIRS:
        out = OUTPUT_DIR / f'gmm_{x_name.replace("/", "_")}_vs_{y_name.replace("/", "_")}.png'
        plot_pair(
            features, gmm, weights,
            x_name, y_name, x_label, y_label, x_clip, y_clip, out,
        )

    print(f'\nDone. Plots written to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
