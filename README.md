# Noisy Regulatory Updating and Persistent Fragmentation in Crypto-Asset Anti-Money Laundering

This repository contains the numerical reproduction code for the paper:

**“Noisy Regulatory Updating and Persistent Fragmentation in Crypto-Asset Anti-Money Laundering”**

The paper studies a stylized discrete-time dynamic model in which regulators update AML implementation intensity using noisy local signals, VASP compliance responds to implementation intensity, and illicit activity relocates across jurisdictions in response to relative regulatory and compliance conditions.

The current revision uses a single reproduction script. Running it reproduces the numerical analyses and figures used in the revised manuscript.

## Repository structure

```text
adaptive-regulation-crypto-aml/
├── LICENSE
├── README.md
├── requirements.txt
└── src/
    └── irfa_revision_numerical_analysis.py
```

## Reproduced results

The script reproduces:

- the baseline dynamic path used for Figure 2;
- the long-horizon fragmentation analysis used for Figure 3;
- the mechanism-ablation results;
- the strict no-spatial-channel counterfactual;
- the observation-noise sensitivity analysis;
- the relocation-responsiveness sensitivity analysis;
- the regulatory-adjustment-speed sensitivity analysis;
- Figure 4 and its component panels;
- the smooth-versus-fixed-step updating comparison used for Appendix Figure C1;
- the window-by-window long-horizon summaries;
- the 100-period block summaries; and
- the reproduction check for the earlier t = 60–120 baseline results.

The main numerical design is:

- J = 5 jurisdictions;
- T = 2000 periods;
- M = 500 Monte Carlo replications;
- baseline random seed = 123;
- principal long-horizon evaluation window: t = 1001,...,2000.

Common random numbers are used across the counterfactual exercises.

## Requirements

Python 3.9 or later is recommended.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The code requires:

- NumPy
- pandas
- Matplotlib

## Usage

From the repository root, run:

```bash
python src/irfa_revision_numerical_analysis.py
```

The script creates an output directory named:

```text
irfa_revision_results/
```

The directory contains the numerical summaries, figure files, and a run manifest documenting the main simulation settings.

## Interpretation

The numerical exercises are illustrative rather than empirically calibrated. The revised paper does not interpret positive finite-horizon cross-sectional variance alone as evidence of persistence. Instead, the long-horizon analysis examines whether post-transient cross-jurisdictional dispersion shows a tendency to decay toward zero and how its magnitude and composition depend on observation noise, permanent jurisdictional heterogeneity, regulatory updating, and endogenous relocation.

The mechanism-ablation exercises are counterfactual comparisons. They should not be interpreted as an additive variance decomposition.

## Reproducibility

The simulation design uses deterministic random seeds. With the package versions and Python environment held fixed, the numerical results should be reproducible up to ordinary numerical and rendering differences across platforms.

## License

See `LICENSE`.
