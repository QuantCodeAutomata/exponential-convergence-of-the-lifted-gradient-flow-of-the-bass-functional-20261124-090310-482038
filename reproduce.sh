#!/usr/bin/env bash
# Runs the test suite and every experiment in the configuration documented in
# results/RESULTS.md. Each exp.* script overwrites its own files in results/.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== Unit and regression tests =="
python -m pytest -m "not market_data"

echo "== exp_1: Bass functional properties =="
python -m exp.exp1

echo "== exp_2: Lipschitz continuity of the gradient =="
python -m exp.exp2

echo "== exp_3: strong convexity (Theorem 1.1) =="
python -m exp.exp3

echo "== exp_4: exponential convergence of the gradient flow (Theorem 1.2) =="
python -m exp.exp4

echo "== exp_5: initial-value error of the Bass martingale (Section 5) =="
python -m exp.exp5

echo "Done. See results/RESULTS.md for the comparison with the paper."
