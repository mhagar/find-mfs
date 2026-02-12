# `find-mfs`: Accurate mass ➜ Molecular Formulae

[![CI](https://github.com/mhagar/find-mfs/actions/workflows/ci.yml/badge.svg)](https://github.com/mhagar/find-mfs/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/find-mfs)](https://pypi.org/project/find-mfs/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`find-mfs` is a simple Python package for finding 
molecular formulae candidates which fit some given mass (+/- an error window).
It implements Böcker & Lipták's algorithm for efficient formula finding, as 
implemented in SIRIUS. 

`find-mfs` also implements other methods 
for filtering the MF candidate lists:
- **Octet rule**
- **Ring/double bond equivalents (RDBE's)**
- **Predicted isotope envelopes**, generated using Łącki and Startek's algorithm
  as implemented in `IsoSpecPy`

## Motivation:
I needed to perform mass decomposition and, shockingly, I could not find a Python library for it 
(despite being a routine process). `find-mfs` is intended to be used by anyone looking to incorporate
molecular formula finding into their Python project.

## Installation
```commandline
pip install find-mfs
```

## Example Usage:

**Simple queries**
```python
# For simple queries, one can use this convenience function
from find_mfs import find_chnops

find_chnops(
    mass=613.2391,         # Novobiocin [M+H]+ ion; C31H37N2O11+
    charge=1,              # Charge should be specified - electron mass matters
    error_ppm=5.0,         # Can also specify error_da instead
                           # --- FORMULA FILTERS ----
    check_octet=True,      # Candidates must obey the octet rule
    filter_rdbe=(0, 20),   # Candidates must have 0 to 20 ring/double-bond equivalents
    max_counts='C*H*N*O*P0S2'      # Element constraints: unlimited C/H/N/O,
                                   # No phosphorous atoms, up to two sulfurs.
)
```
Output:
```
FormulaSearchResults(query_mass=613.2391, n_results=38)

Formula                   Error (ppm)     Error (Da)      RDBE
----------------------------------------------------------------------
[C6H25N30O4S]+                     -0.12       0.000073       9.5
[C31H37N2O11]+                      0.14       0.000086      14.5
[C14H29N24OS2]+                     0.18       0.000110      12.5
[C16H41N10O11S2]+                   0.20       0.000121       1.5
[C29H33N12S2]+                     -0.64       0.000392      19.5
... and 33 more
```
**Batch Queries**
```python
# If processing many masses, it's better to instantiate a FormulaFinder object
from find_mfs import FormulaFinder

finder = FormulaFinder()
finder.find_formulae(
    mass=613.2391,         # Novobiocin [M+H]+ ion; C31H37N2O11+
    charge=1,              
    error_ppm=5.0,         
    # ... etc
)
```
**Including Isotope Envelope Information**

If an isotope envelope is available, the candidate list can be dramatically
reduced. 

```python
import numpy as np

# STEP 1: Retrieve isotope envelope from experimental data
observed_envelope = np.array(
    [  #  m/z    , relative intsy.
        [613.2397,    1.00],
        [614.2429,    0.35],
        [615.2456,    0.10],
    ]
)

# STEP 2: define isotope matching parameters
from find_mfs import SingleEnvelopeMatch
iso_config = SingleEnvelopeMatch(
    envelope=observed_envelope,     # np.ndarray with an m/z column and an intensity column
    mz_tolerance_da=0.005,          # Tolerance for aligning isotope signals. Should be very generous. Can also use mz_tolerance_ppm
    minimum_rmse=0.05,              # Default is 0.05, i.e. instrument reproduces isotope envelope w/ 5% fidelity
)

# STEP 3: include isotope matching parameters when performing a search
from find_mfs import FormulaFinder
finder = FormulaFinder()
finder.find_formulae(
    mass=613.2391,         # Novobiocin [M+H]+ ion; C31H37N2O11+
    charge=1,              # Charge should be specified - electron mass matters
    error_ppm=3.0,         # Can also specify error_da instead
                           # --- FORMULA FILTERS ----
    check_octet=True,      # Candidates must obey the octet rule
    filter_rdbe=(0, 20),   # Candidates must have 0 to 20 ring/double-bond equivalents
    max_counts={
        'P': 0,            # Candidates must not have any phosophorous atoms
        'S': 2,            # Candidates can have up to two sulfur atoms
    },
    isotope_match=iso_config,
)
```
Output:
```
FormulaSearchResults(query_mass=613.2391, n_results=5)

Formula                   Error (ppm)     Error (Da)      RDBE       Iso. Matches   Iso. RMSE 
------------------------------------------------------------------------------------------------------
[C31H37N2O11]+                      0.14       0.000086      14.5           3/3    0.0121
[C23H41N4O13S]+                    -0.92       0.000565       5.5           3/3    0.0478
[C24H37N8O9S]+                      1.26       0.000772      10.5           3/3    0.0311
[C32H33N6O7]+                       2.32       0.001424      19.5           3/3    0.0230
[C25H33N12O5S]+                     3.44       0.002110      15.5           3/3    0.0146
```

### Jupyter Notebook:
See [this Jupyter notebook](docs/basic_usage.ipynb) for more thorough examples/demonstrations

---
**If you use this package, make sure to cite:**
- [Böcker & Lipták, 2007](https://link.springer.com/article/10.1007/s00453-007-0162-8) - this package uses their algorithm for formula finding...
    - ...as implemented in SIRIUS: [Böcker et. al., 2008](https://academic.oup.com/bioinformatics/article/25/2/218/218950)
- [Łącki, Valkenborg & Startek 2020](https://pubs.acs.org/doi/10.1021/acs.analchem.0c00959) - this package uses IsoSpecPy to quickly simulate isotope envelopes
- [Gohlke, 2025](https://zenodo.org/records/17059777) - this package uses `molmass`, which provides very convenient methods for handling chemical formulae


## Contributing

Contributions are welcome. Here's a list of features I feel should be implemented eventually.
The bold items are what I'm currently working on.
- ~~Statistics-based isotope envelope fitting~~
- ~~Fragmentation constraints~~
- **Bayesian formula candidate ranking**
- Element ratio constraints
- GUI app

## License

This project is distributed under the GPL-3 license.
