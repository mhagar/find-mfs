"""
Generate the bundled two-model default chemical prior.

Fits the per-class GMMs (halogenated vs halogen-free) on the COCONUT corpus and
serializes the parameters into `find_mfs/scoring/_coconut_gmm.py`,
 i.e. the GMM parameters loaded by `FormulaScorer.default()`

Run after changing the feature layout, the tau percentile, or the corpus:
    uv run python scripts/generate_default_gmm.py
"""
from __future__ import annotations

import json
from pathlib import Path

from find_mfs.scoring.scorer import FormulaScorer, _CLASSES

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "COCONUT_molecular_formulae.txt"
OUT = ROOT / "find_mfs" / "scoring" / "_coconut_gmm.py"

# Components per class. The halogenated subset is smaller/less diverse; adjust
# here (or BIC-select via scorer.select_n_components on each subset) if needed.
N_COMPONENTS = 25
RANDOM_STATE = 42
TAU_PERCENTILE = 0.5

_HEADER = '''"""
Pre-trained two-model chemical-prior parameters.

Corpus:        COCONUT natural-products database
Models:        one GMM per composition class -- 'halogen' (contains Cl/Br/I) and
               'halofree' (everything else, fluorine included)
Components:    k={k} per class
Gate floor:    tau = {p}th-percentile of each class's training log-densities
Feature order: see find_mfs.scoring.scorer (H/C, O/C, N, S, P, Cl, Br, I,
               RDBE, RDBE/C, n_heteroatom_types)

Regenerate with scripts/generate_default_gmm.py. Loaded via FormulaScorer.default().
"""
DEFAULT_GMM_PARAMS = '''


def main() -> None:
    if not CORPUS.exists():
        raise SystemExit(f"Corpus not found: {CORPUS}")

    formulae = CORPUS.read_text().splitlines()
    print(f"Fitting on {len(formulae):,} formulae from {CORPUS.name} ...")
    scorer = FormulaScorer().fit(
        formulae,
        n_components=N_COMPONENTS,
        random_state=RANDOM_STATE,
        tau_percentile=TAU_PERCENTILE,
    )

    params: dict[str, dict | None] = {}
    for cls in _CLASSES:
        gmm = scorer._models[cls]
        if gmm is None:
            params[cls] = None
            print(f"  {cls}: skipped (too few formulae)")
            continue
        params[cls] = {
            "n_components": gmm.n_components,
            "weights": gmm.weights_.tolist(),
            "means": gmm.means_.tolist(),
            "covariances": gmm.covariances_.tolist(),
            "tau": scorer._taus[cls],
            "scale": scorer._scales[cls],
        }
        print(
            f"  {cls}: k={gmm.n_components}, tau={scorer._taus[cls]:.3f}, "
            f"scale={scorer._scales[cls]:.3f}"
        )

    # json.dumps output is valid Python here (dicts/lists/floats/strings only).
    body = json.dumps(params, indent=1)
    OUT.write_text(_HEADER.format(k=N_COMPONENTS, p=TAU_PERCENTILE) + body + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
