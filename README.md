# Exponential Convergence of the Lifted Gradient Flow of the Bass Functional

Numerical replication of selected results from *"Exponential Convergence of
the Lifted Gradient Flow of the Bass Functional"*.

This repository verifies, on a concrete synthetic instance obeying the
standing assumptions of Section 1.1 (a pair of laws `nu0 <=_cx nu1` with
`nu1` compactly supported, `supp(nu0)` in the interior of `conv(supp nu1)`),
the following results of the paper:

* **exp_1** -- finiteness, attainment via Brenier optimal transport, Frechet
  differentiability of the Bass functional `V`, and the minimizer
  characterization `g(Z*) = 0` (Properties 1.1, 1.2, 1.3, 1.6).
* **exp_2** -- the Gaussian first-order remainder bound (Lemma 2.3), the
  Brenier-potential identity and Lipschitz property (Lemma 2.4), the
  `M`-Lipschitz continuity of the gradient `g` (Lemma 2.5), and the convexity
  of the auxiliary functional `F(Z) = (M/2)||Z||_2^2 - V(Z)`.
* **exp_3** -- the strong-convexity constant `m_R` of Theorem 1.1, the
  directional-derivative lower bound of Lemma 3.1, and the strong-convexity
  inequality itself.
* **exp_4** -- exponential convergence of the gradient flow
  `dZ_t/dt = -g(Z_t)` to the minimizer `Z*` (Theorem 1.2): decay of
  `||Z_t - Z*||_2` at rate `m` and of `V(Z_t) - V(Z*)` at rate `2m`, together
  with conservation of the mean and boundedness of `||Z_t||_inf`.
* **exp_5** -- the error bound on the initial value of the Bass martingale
  built from a gradient-flow iterate `Z_t` (Section 5): decay of
  `||g(Z_t)||_2` and convergence of the martingale's initial law to `nu0`.

The paper gives no numerical example of `(nu0, nu1)`; every experiment uses
the same synthetic instance, constructed in `data/generate_synthetic.py` from
an explicit sufficient condition for the convex order (an independent,
zero-mean, bounded perturbation), and documented there together with the
standing assumptions it is built to satisfy.

No market data, external APIs, or API keys are used anywhere in this
repository.

## Layout

```
src/    Bass functional, gradient, optimal transport, gradient flow,
        martingale and strong-convexity modules, and src/config.py with
        every experiment's parameters.
exp/    One script per experiment (exp1 .. exp5), runnable as
        `python -m exp.exp1` etc.
data/   data/generate_synthetic.py builds the synthetic (nu0, nu1) instance
        and the Gaussian/Brownian samples used throughout.
tests/  Unit tests for src/ components and regression tests for each
        experiment's qualitative claims.
results/ Tables, logs and plots written by the experiment scripts, and
        results/RESULTS.md comparing them with the paper.
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Running the experiments

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4            # reduced run, see results/RESULTS.md
python -m exp.exp4 --full     # paper-scale number of gradient-flow steps
python -m exp.exp5
python -m exp.exp5 --full
```

Each script writes its tables, logs and figures directly to `results/` and
prints a short summary. `reproduce.sh` runs the test suite and every
experiment in the configuration described in `results/RESULTS.md`.

```bash
./reproduce.sh
```
