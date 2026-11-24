"""Parameters for the numerical verification of the Bass-functional results.

All parameters below are module-level constants grouped by experiment. Every
constant carries a short comment stating its meaning, its unit, and whether it
is taken from the paper ("Exponential Convergence of the Lifted Gradient Flow
of the Bass Functional") or left to the implementation because the paper does
not fix a numerical value for it ("not specified in the paper").

The paper verifies its properties and theorems for *any* pair of laws (nu0,
nu1) obeying the standing assumptions of Section 1.1; it gives no specific
numerical example. The distributions, sample sizes, and random seeds below are
therefore our own choice of a concrete, assumption-compliant instance, used
identically across every experiment unless stated otherwise.
"""

# ---------------------------------------------------------------------------
# Shared synthetic model (paper Section 1.1: standing assumptions on nu0, nu1)
# ---------------------------------------------------------------------------
DIMENSIONS = (1, 2)            # dimensions d tested for generality (paper proves the rate "in arbitrary dimension")
SUPPORT_RADIUS = 1.0           # M = max_{y in supp nu1} |y|; nu1 is uniform on the ball of this radius (Section 1.1, item 1)
N_Y_SAMPLES = 600              # number of i.i.d. draws approximating nu1 (not specified in the paper; our discretization)
N_X_ATOMS = 30                 # number of atoms of nu0 = number of K-means clusters of the nu1 sample (not specified in the paper)
K_GAMMA = 6                    # independent Gamma replicates per atom used to estimate E[.|X] (not specified in the paper)
BASE_SEED = 0                  # root random seed for reproducibility (not specified in the paper)

# ---------------------------------------------------------------------------
# exp_1: finiteness, attainment, Frechet differentiability, minimizer characterization
# ---------------------------------------------------------------------------
EXP1_N_TEST_Z = 12             # number of random Z in H tested for the finiteness bound (Property 1.1)
EXP1_FD_DIRECTIONS = 10        # number of random directions W for the finite-difference check of g(Z) (Property 1.3)
EXP1_FD_STEPS = (1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3)  # step sizes h for the finite-difference ratio (Property 1.3)
EXP1_FD_K_GAMMA = 60           # Gamma replicates per atom used only for the differentiability check: a
                               # discrete-OT re-solve at each perturbed Z is a combinatorial (piecewise-affine)
                               # function of Z, so the finite-difference error is U-shaped in h (truncation error
                               # for large h, re-matching noise for very small h); more replicates lower this
                               # noise floor (not specified in the paper, chosen for a visible minimum within EXP1_FD_STEPS)
EXP1_MINIMIZER_STEPS = 400     # gradient-flow Euler steps used to approximate the minimizer Z* (Property 1.4/1.6)
EXP1_MINIMIZER_DT = 0.05       # Euler step size dt for the flow that locates Z* (not specified in the paper)

# ---------------------------------------------------------------------------
# exp_2: Lipschitz continuity of g (Lemmas 2.3, 2.4, 2.5) and convexity of F
# ---------------------------------------------------------------------------
EXP2_LEMMA23_N_MC = 20000      # Monte Carlo sample size for the Gaussian-shift bound of Lemma 2.3
EXP2_LEMMA23_H_GRID = (0.1, 0.3, 0.6, 1.0, 1.5, 2.0)  # test offsets |h| for Lemma 2.3 (not specified in the paper)
EXP2_N_RANDOM_PAIRS = 40       # number of random (Z, W) pairs tested for Lemma 2.5 / convexity of F
EXP2_N_W1_CHECK = 4000         # number of resampled target points used to check that Y_Z has law nu1 (Property 1.2)

# ---------------------------------------------------------------------------
# exp_3: strong convexity (Theorem 1.1) and the directional-derivative bound (Lemma 3.1)
# ---------------------------------------------------------------------------
EXP3_B_DIRECTIONS = 64         # number of random unit directions e used in the Monte Carlo minimization defining b (Section 3.1)
EXP3_R_VALUES = (0.5, 0.8, 1.2)  # L^infinity bounds R for which m_R is evaluated (Theorem 1.1); not specified in the paper
EXP3_N_SC_PAIRS = 30           # number of random (Z, W) pairs with E[Z]=E[W], ||.||_inf <= R, used to test strong convexity
EXP3_LEMMA31_RHO_FACTOR = 2.0  # rho is set to LEMMA31_RHO_FACTOR * M(2E|Q|+sqrt(d))/b, keeping rho above the Lemma 3.1 threshold
EXP3_LEMMA31_SIGMA = 1.0       # sigma in (0,1] used in Lemma 3.1 (paper allows any such value; we use the boundary case)
EXP3_LEMMA31_N_MC = 20000      # Monte Carlo sample size for the directional-derivative expectation in Lemma 3.1

# ---------------------------------------------------------------------------
# exp_4: exponential convergence of the gradient flow (Theorem 1.2)
# ---------------------------------------------------------------------------
EXP4_DT = 0.05                 # Euler step size dt for the gradient flow dZ/dt = -g(Z) (not specified in the paper)
EXP4_N_STEPS_PAPER = 800       # number of Euler steps in the paper configuration (not specified in the paper; chosen for clear exponential decay)
EXP4_N_STEPS_RUN = 300         # number of Euler steps actually executed (reduced to keep the full study within the time available)
EXP4_Z0_SCALE = 1.6            # initial vector Z0 = EXP4_Z0_SCALE * x_atoms (a bounded, L^infinity-valued perturbation of X)
EXP4_FIT_SKIP_FRACTION = 0.15  # fraction of early iterations excluded from the log-linear decay-rate fit (transient removal)
EXP4_FLOW_K_GAMMA = 60         # Gamma replicates per atom for the gradient-flow steps themselves (not specified in the paper): larger than the shared K_GAMMA because the flow reuses one fixed Gamma draw along the whole trajectory (common random numbers), so the finite-K Monte Carlo noise of the discrete-OT estimator of both g and V, not just a bias, must be small enough that the flow tracks the population ODE of Theorem 1.2 and the V(Z_t) decay-rate fit is not swamped by sampling noise

# ---------------------------------------------------------------------------
# exp_5: error in the initial value of the Bass martingale (Section 5)
# ---------------------------------------------------------------------------
EXP5_DT = 0.05                 # Euler step size dt for the gradient flow reused from Theorem 1.2 (shared with exp_4)
EXP5_N_STEPS_PAPER = 800       # paper-scale number of flow steps from which Bass martingales are built (matches EXP4_N_STEPS_PAPER)
EXP5_N_STEPS_RUN = 300         # reduced number of flow steps actually executed (matches EXP4_N_STEPS_RUN)
EXP5_EVAL_TIMES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)  # stopping times t at which the martingale error is evaluated
EXP5_N_BRIDGE_PATHS = 2000     # inner Monte Carlo sample size for the Brownian-bridge conditional expectation defining M_s (Section 5)
EXP5_BRIDGE_S_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)  # times s in [0,1] at which the Bass martingale M_s is evaluated
EXP5_N_OUTER_PATHS = 40        # number of outer Brownian paths W_s along which M_s is estimated (not specified in the paper)
