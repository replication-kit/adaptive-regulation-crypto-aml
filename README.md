# Adaptive Regulation and Soft Lock-In in Crypto-Asset–Based Anti–Money Laundering

This repository contains the simulation code used in the paper  
“Adaptive Regulation and Soft Lock-In in Crypto-Asset–Based Anti–Money Laundering”.

The code implements a stochastic dynamic model in which regulatory authorities
adapt enforcement intensity based on noisy detection outcomes, while illicit
financial flows relocate endogenously across jurisdictions. The simulations
illustrate how adaptive regulation can generate persistent regulatory gaps
and regime-dependent dynamics.

## Files

- `src/Figure2.py`  
  Reproduces Figure 2 in the paper.  
  The figure presents regime maps over the adjustment step (Δ) and observation
  noise (σ):
  - Panel A: regime multiplicity measured by a dispersion-based lock-in index  
  - Panel B: probability of severe lock-in

- `src/Figure3.py`  
  Reproduces Figure 3 in the paper.  
  The figure shows representative time-series dynamics of detection probabilities
  and criminal fund concentration for parameter points corresponding to
  stable, multiplicity, and severe lock-in regimes identified in Figure 2.

- `src/Figurec.py`  
  Reproduces Appendix Figure C.  
  This figure reports sensitivity of the lock-in patterns to the mobility
  parameter governing the relocation of illicit financial flows.

- `src/Tablec.py`  
  Produces Appendix Table C.  
  The table reports robustness of the main results to an alternative, smoothed
  regulatory update rule, focusing on the lock-in index and the probability of
  severe lock-in at representative parameter points.

## Requirements

The code was tested with Python 3.9+ and requires the following packages:

numpy
matplotlib

To install dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts from the repository root:

```bash
python src/Figure2.py
python src/Figure3.py
python src/FigureC.py
python src/TableC.py
```

## Notes

All results are based on Monte Carlo simulations.

Numerical values may vary slightly across runs due to randomness.

The code is intended to illustrate qualitative mechanisms and regime
patterns rather than to provide a precise quantitative calibration.

## License

This code is provided for academic and non-commercial use.