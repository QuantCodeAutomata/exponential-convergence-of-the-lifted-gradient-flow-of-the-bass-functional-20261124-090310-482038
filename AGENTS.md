# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A numerical replication of five results from *"Exponential Convergence of the
Lifted Gradient Flow of the Bass Functional"* (Danila Litvinov,
arXiv:2610.40404, math.PR, HSE University):

* `exp_1` — finiteness, attainment via Brenier optimal transport, and Frechet
  differentiability of the Bass functional `V` (Properties 1.1-1.3), and the
  minimizer characterization `g(Z*) = 0` (Property 1.6).
* `exp_2` — the Gaussian first-order remainder bound (Lemma 2.3), the
  Brenier-potential identity and Lipschitz property (Lemma 2.4), the
  `M`-Lipschitz continuity of the gradient `g` (Lemma 2.5), and convexity of
  `F(Z) = (M/2)||Z||_2^2 - V(Z)`.
* `exp_3` — the strong-convexity constant `m_R` and inequality of Theorem 1.1,
  and the directional-derivative lower bound of Lemma 3.1.
* `exp_4` — exponential convergence of the gradient flow `dZ_t/dt = -g(Z_t)`
  to the minimizer `Z*` (Theorem 1.2).
* `exp_5` — the error bound on the initial value of the Bass martingale built
  from a gradient-flow iterate (Section 5).

The paper proves its results for any pair of laws `(nu0, nu1)` satisfying the
standing assumptions of Section 1.1 and gives no numerical example; every
experiment uses one explicit, assumption-compliant synthetic instance built in
`data/generate_synthetic.py`. No market data, external APIs, or API keys are
used anywhere in this repository.

## Layout

```
src/      Flat modules: bass_functional.py, optimal_transport.py,
          gradient_flow.py, strong_convexity.py, martingale.py, and
          config.py (every experiment's parameters, as module-level
          constants with a comment citing the paper or "not specified").
exp/      One script per experiment, run as `python -m exp.exp1` .. exp5.
data/     generate_synthetic.py builds the shared synthetic (nu0, nu1)
          instance and the Gaussian/Brownian samples used throughout.
tests/    pytest tests for src/ components and regression tests for each
          experiment's qualitative claims.
results/  Tables (JSON), figures (PNG) and RESULTS.md written directly by
          the experiment scripts; no per-experiment subdirectories.
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests (no market data is used, so there is no `market_data` marker to skip,
but the invocation below is the one used elsewhere in this project family):

```bash
python -m pytest -m "not market_data"
```

Experiments (reduced configuration, then the paper-scale configuration for
`exp_4`/`exp_5`):

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp4 --full
python -m exp.exp5
python -m exp.exp5 --full
```

`./reproduce.sh` runs the test suite and the reduced configuration of every
experiment in one command.

## Conventions

* All experiment parameters live only in `src/config.py` as module-level
  `UPPER_CASE` constants, each with a comment giving its meaning, unit, and
  source (a paper section/equation, or "not specified in the paper").
* Data preparation lives only in `data/`: `data/generate_synthetic.py`
  generates the synthetic `(nu0, nu1)` instance; nothing in `src/` or `exp/`
  generates data of its own.
* `exp/*.py` scripts call only the public functions of `src/`, using their
  documented signatures and return keys; they do not implement private
  copies of the model, the optimal-transport solver, or the gradient flow.
* Rerunning an experiment script overwrites its files in `results/`. After a
  rerun, the tables and verdicts in `results/RESULTS.md` must be checked
  against the newly written files and updated by hand; they are not
  regenerated automatically.
* No API keys are required or read by this repository.
