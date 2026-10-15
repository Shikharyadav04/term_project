"""
BT-JRNN Model Order Reduction
=============================
Implementation of:
  "A Model Order Reduction Technique Based on Balanced Truncation Method
   and Artificial Neural Networks" (ICMSAO 2019)

Pipeline
--------
  Plant G(s) (6th order)
     |-- Balanced Truncation (BT)  -------------------> G_BT(s)
     |-- JRNN (neural net) --> IV estimation ---------> G_IV-JRNN(s)
  Cascade:  G_BT-JRNN(s) = G_BT(s) * G_IV-JRNN(s)          (eq. 2)

Finally: Bode-plot MSE table (magnitude / phase) + time-domain MSE,
and a second table with the step-response ISE / RISE / IAE / ITAE
for  JRNN , BT , BT-JRNN.

Needs: numpy, scipy, matplotlib
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg, optimize

warnings.filterwarnings("ignore", message="Badly conditioned filter coefficients")

# =====================================================================
#                      >>>  USER SETTINGS (EDIT HERE)  <<<
# =====================================================================

# ---- 1) PLANT TRANSFER FUNCTION  G(s) = NUM(s) / DEN(s) -------------
# Coefficients in DESCENDING powers of s.  (Paper example, eq. 3)
PLANT_NUM = [1, -20, 180, -840, 1680]                    # s^4 - 20s^3 + 180s^2 - 840s + 1680
PLANT_DEN = [1, 22.5, 231, 1310, 3960, 5040, 1680]       # s^6 + 22.5s^5 + 231s^4 + ... + 1680

# ---- 2) INPUT SIGNAL used to train / test the neural network --------
N_SAMPLES   = 8000      # total number of random samples (paper: 8000)
INPUT_MIN   = 0.0       # random input range (paper: 0 to 4)
INPUT_MAX   = 4.0
HOLD_MIN    = 5         # each random level is held for HOLD_MIN..HOLD_MAX samples
HOLD_MAX    = 20        #   (gives the step-like input seen in Fig. 4)
TS          = 0.1       # sample time [s]
RANDOM_SEED = 0

# >>> To test your own input, set this to a function / array and it will be
# >>> used for the TEST set instead of the random one. Examples:
#       CUSTOM_TEST_INPUT = lambda t: 2 + np.sin(2*np.pi*0.5*t)
#       CUSTOM_TEST_INPUT = lambda t: 4*(np.mod(t, 5) < 2.5)     # square wave
CUSTOM_TEST_INPUT = None

# ---- 3) REDUCTION ORDERS --------------------------------------------
BT_ORDER_CASCADE  = 2   # order of BT model inside BT-JRNN   (paper eq. 4: 2nd order)
IV_ORDER_CASCADE  = 2   # order of IV-JRNN model in cascade  (paper eq. 5: 2nd order)
#   => BT-JRNN order = 2 + 2 = 4  (paper eq. 6)
FAIR_ORDER        = BT_ORDER_CASCADE + IV_ORDER_CASCADE   # JRNN and BT alone use this order
BT_METHOD         = "matchdc"   # "matchdc" (residualization, = MATLAB balred default,
                                #  reproduces paper eq. 4)  or  "truncate"

# ---- 4) JRNN SETTINGS -----------------------------------------------
HIDDEN_NEURONS = 7      # paper: 7 hidden neurons
TRAIN_FRAC, VAL_FRAC = 0.50, 0.25      # 50 % train / 25 % validation / 25 % test
JRNN_MAX_ITERS = 300    # Levenberg-Marquardt iterations (stage 1); stage 2 uses max_iters//20 rounds
JRNN_RESPONSE  = "one_step"   # "one_step": JRNN gets the plant's past outputs (as in paper Fig. 4)  [default]
                              # "free_run": JRNN feeds its OWN outputs back (harder; can drift)
JRNN_FREE_RUN_FINETUNE = False  # extra (slow) training stage on multi-step error; only useful for "free_run"

# ---- 5) IV ESTIMATION / EVALUATION ----------------------------------
IV_ITERATIONS  = 40
IV_PREFILTER_P = 3.0           # initial state-variable-filter pole
W_MIN, W_MAX, N_FREQ = 1.0, 100.0, 500   # Bode comparison range [rad/s] (paper: 1-100)
MAG_UNITS = "linear"          # magnitude MSE on |G(jw)| ("linear") or on 20*log10|G(jw)| ("dB")
PAPER_SCALE = 10.0             # ISE, IAE, ITAE are multiplied by this (same convention as your reference code)
DT_EVAL     = 0.005            # time step used to integrate the step-response error indices
SAVE_FIGS = True               # save PNGs next to this script
# =====================================================================


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def fmt_tf(num, den, name):
    """Pretty-print a transfer function."""
    def poly(c):
        n = len(c) - 1
        terms = []
        for i, v in enumerate(c):
            p = n - i
            if abs(v) < 1e-12:
                continue
            s = f"{v:.4g}"
            terms.append(s if p == 0 else (f"{s}s" if p == 1 else f"{s}s^{p}"))
        return " + ".join(terms).replace("+ -", "- ") or "0"
    n_, d_ = poly(np.atleast_1d(num)), poly(np.atleast_1d(den))
    bar = "-" * max(len(n_), len(d_))
    print(f"\n{name}\n  {n_}\n  {bar}\n  {d_}")


def make_random_input(n, rng, lo, hi, hold_min, hold_max):
    """Random piece-wise constant signal in [lo, hi]."""
    u = np.empty(n)
    i = 0
    while i < n:
        h = rng.integers(hold_min, hold_max + 1)
        u[i:i + h] = rng.uniform(lo, hi)
        i += h
    return u


def simulate_tf(num, den, u, ts):
    """Exact (ZOH) simulation of a continuous TF for a sampled input."""
    num = np.atleast_1d(num).astype(float)
    den = np.atleast_1d(den).astype(float)
    A, B, C, D = signal.tf2ss(num, den)
    Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), ts, method="zoh")
    _, y, _ = signal.dlsim((Ad, Bd, Cd, Dd, ts), u.reshape(-1, 1))
    return y.ravel()


def trim_tf(num, den, tol=1e-9):
    num = np.trim_zeros(np.atleast_1d(num), "f")
    return (num if len(num) else np.array([0.0])), np.atleast_1d(den)


# ---------------------------------------------------------------------
# 1) Balanced truncation
# ---------------------------------------------------------------------
def balanced_reduction(num, den, order, method="matchdc"):
    A, B, C, D = signal.tf2ss(num, den)
    if np.any(np.linalg.eigvals(A).real >= 0):
        raise ValueError("Plant must be stable for balanced truncation.")
    Wc = linalg.solve_continuous_lyapunov(A, -B @ B.T)
    Wo = linalg.solve_continuous_lyapunov(A.T, -C.T @ C)
    Lc = linalg.cholesky(Wc, lower=True)
    Lo = linalg.cholesky(Wo, lower=True)
    U, S, Vt = linalg.svd(Lo.T @ Lc)
    Sm12 = np.diag(S ** -0.5)
    T = Lc @ Vt.T @ Sm12
    Ti = Sm12 @ U.T @ Lo.T
    Ab, Bb, Cb = Ti @ A @ T, Ti @ B, C @ T
    r = order
    A11, A12, A21, A22 = Ab[:r, :r], Ab[:r, r:], Ab[r:, :r], Ab[r:, r:]
    B1, B2, C1, C2 = Bb[:r], Bb[r:], Cb[:, :r], Cb[:, r:]
    if method == "truncate":
        Ar, Br, Cr, Dr = A11, B1, C1, D
    else:  # matchdc: singular perturbation / residualization -> keeps DC gain
        X = linalg.solve(A22, np.hstack([A21, B2]))
        Ar = A11 - A12 @ X[:, :r]
        Br = B1 - A12 @ X[:, r:]
        Cr = C1 - C2 @ linalg.solve(A22, A21)
        Dr = D - C2 @ linalg.solve(A22, B2)
    n, d = signal.ss2tf(Ar, Br, Cr, Dr)
    return trim_tf(n[0], d), S


# ---------------------------------------------------------------------
# 2) Jordan Recurrent Neural Network (inputs: x_t, y_{t-1}, y_{t-2})
# ---------------------------------------------------------------------
class JRNN:
    """
    y_t = Wy . sigmoid(Wn*x_t + Un*y_{t-1} + Kn*y_{t-2} + bn) + by     (paper eq. 1)
    hidden: sigmoid, output: linear.
    """
    def __init__(self, h=7, seed=0):
        self.h = h
        rng = np.random.default_rng(seed)
        self.theta = rng.normal(0, 0.5, 5 * h + 1)

    def _unpack(self, th):
        h = self.h
        Wn, Un, Kn, bn, Wy = (th[i * h:(i + 1) * h] for i in range(5))
        by = th[5 * h]
        return Wn, Un, Kn, bn, Wy, by

    def _predict(self, th, x, y1, y2):
        Wn, Un, Kn, bn, Wy, by = self._unpack(th)
        z = np.outer(x, Wn) + np.outer(y1, Un) + np.outer(y2, Kn) + bn
        z = np.clip(z, -50, 50)
        H = 1.0 / (1.0 + np.exp(-z))
        return H @ Wy + by

    def one_step(self, x, y, th=None):
        """Predict y_t from x_t and the *given* previous outputs."""
        th = self.theta if th is None else th
        y1 = np.r_[0.0, y[:-1]]
        y2 = np.r_[0.0, 0.0, y[:-2]]
        return self._predict(th, x, y1, y2)

    def free_run(self, x):
        """Recursive simulation: JRNN feeds its OWN outputs back."""
        y = np.zeros(len(x))
        for t in range(len(x)):
            y1 = y[t - 1] if t >= 1 else 0.0
            y2 = y[t - 2] if t >= 2 else 0.0
            y[t] = self._predict(self.theta, x[t:t + 1], np.array([y1]), np.array([y2]))[0]
        return y

    def _free_run_segments(self, th, xs, y0):
        """Vectorised free-run over many segments. xs: (S,L), y0: (S,2) = [y_{-1}, y_{-2}]."""
        S, L = xs.shape
        out = np.zeros((S, L))
        y1, y2 = y0[:, 0].copy(), y0[:, 1].copy()
        for t in range(L):
            yt = self._predict(th, xs[:, t], y1, y2)
            out[:, t] = yt
            y2, y1 = y1, yt
        return out

    def train(self, x_tr, y_tr, x_va, y_va, max_iters=300, seg_len=100, free_run_stage=True):
        """
        Stage 1: Levenberg-Marquardt on the one-step-ahead error (standard JRNN training).
        Stage 2: Levenberg-Marquardt on the multi-step (free-run) error, so the network
                 stays accurate when it feeds its OWN outputs back (needed for IV estimation).
        The weights with the best validation error are kept.
        """
        # ---- stage 1
        res1 = lambda th: self.one_step(x_tr, y_tr, th) - y_tr
        th = optimize.least_squares(res1, self.theta, method="lm",
                                    max_nfev=max_iters * (len(self.theta) + 1)).x
        self.theta = th
        val_one = np.mean((self.one_step(x_va, y_va, th) - y_va) ** 2)
        if not free_run_stage:
            return val_one

        # ---- stage 2
        S = (len(x_tr) - 2) // seg_len
        idx = 2 + np.arange(S) * seg_len
        xs = np.stack([x_tr[i:i + seg_len] for i in idx])
        ys = np.stack([y_tr[i:i + seg_len] for i in idx])
        y0 = np.stack([[y_tr[i - 1], y_tr[i - 2]] for i in idx])
        res2 = lambda th_: (self._free_run_segments(th_, xs, y0) - ys).ravel()
        best_th = th.copy()
        best_val = np.mean((self._free_run_full(th, x_va, y_va) - y_va) ** 2)
        for _ in range(3):
            th = optimize.least_squares(res2, th, method="lm",
                                        max_nfev=10 * (len(th) + 1)).x
            v = np.mean((self._free_run_full(th, x_va, y_va) - y_va) ** 2)
            if v < best_val:
                best_val, best_th = v, th.copy()
        self.theta = best_th
        return best_val

    def _free_run_full(self, th, x, y_true=None):
        old = self.theta; self.theta = th
        try:
            return self.free_run(x)
        finally:
            self.theta = old


# ---------------------------------------------------------------------
# 3) Instrumental-Variable continuous-time TF estimation (SRIVC-style)
# ---------------------------------------------------------------------
def iv_estimate(u, y, ts, order, num_deg=None, iters=40, p0=3.0):
    """
    Estimate G(s)=B(s)/A(s) from sampled u,y.  Uses state-variable filters
    1/F(s) to get derivatives and an instrument built from the noise-free
    auxiliary model output (instrumental variable).  F(s) is updated with
    the current A(s) at every iteration (refined IV).
    """
    n = order
    m = n - 1 if num_deg is None else num_deg
    F = np.poly([-p0] * n)                      # initial filter (s+p0)^n
    inst_model = None                            # auxiliary model (num, den)
    theta = None

    def filt(sig, k, Fpoly):
        """s^k / F(s) applied to sig (k <= n)."""
        numk = np.zeros(n + 1); numk[n - k] = 1.0
        nd, dd, _ = signal.cont2discrete((numk, Fpoly), ts, method="zoh")
        return signal.lfilter(nd.ravel(), dd, sig)

    best_theta, best_err = None, np.inf
    for it in range(iters):
        x_aux = y if inst_model is None else simulate_tf(inst_model[0], inst_model[1], u, ts)
        phi_y  = [filt(y,     k, F) for k in range(n - 1, -1, -1)]    # s^{n-1}..s^0
        phi_u  = [filt(u,     k, F) for k in range(m, -1, -1)]        # s^m..s^0
        psi_y  = [filt(x_aux, k, F) for k in range(n - 1, -1, -1)]
        Phi = np.column_stack([-v for v in phi_y] + phi_u)
        Psi = np.column_stack([-v for v in psi_y] + phi_u)
        target = filt(y, n, F)
        skip = 3 * n                                                  # drop filter transient
        try:
            th = np.linalg.solve(Psi[skip:].T @ Phi[skip:], Psi[skip:].T @ target[skip:])
        except np.linalg.LinAlgError:
            break
        a, b = th[:n], th[n:]
        den = np.r_[1.0, a]
        if np.any(np.roots(den).real >= 0):       # keep the model stable
            den = np.poly(-np.abs(np.roots(den).real) + 1j * np.roots(den).imag)
            den = np.real(den)
        inst_model = (b, den)
        F = den
        err = np.mean((simulate_tf(b, den, u, ts) - y) ** 2)
        if err < best_err:
            best_err, best_theta = err, (b.copy(), den.copy())
    return trim_tf(*best_theta)


# ---------------------------------------------------------------------
# 4) Evaluation: Bode MSE + time-domain MSE
# ---------------------------------------------------------------------
def bode_data(num, den, w):
    _, h = signal.freqs(num, den, w)
    mag = 20 * np.log10(np.abs(h)) if MAG_UNITS == "dB" else np.abs(h)
    ph = np.degrees(np.unwrap(np.angle(h)))
    return mag, ph


def bode_mse(ref, mod, w):
    mr, pr = bode_data(*ref, w)
    mm, pm = bode_data(*mod, w)
    pm = pm - 360 * np.round((pm[0] - pr[0]) / 360)      # align 360-degree branch
    return np.mean((mr - mm) ** 2), np.mean((pr - pm) ** 2)


def series(tf1, tf2):
    return np.polymul(tf1[0], tf2[0]), np.polymul(tf1[1], tf2[1])


# ---------------------------------------------------------------------
# 5) Step-response error indices: ISE, RISE, IAE, ITAE
# ---------------------------------------------------------------------
_trapz = getattr(np, "trapezoid", None) or np.trapz


class StepReference:
    """Stores the step response of the original plant and scores reduced models against it."""
    def __init__(self, plant):
        slowest = np.min(np.abs(np.roots(plant[1]).real))
        self.t = np.arange(0, max(100.0, 15.0 / slowest), DT_EVAL)
        _, self.y = signal.step(plant, T=self.t)
        _, h = signal.impulse(plant, T=self.t)
        self.h2 = _trapz(h ** 2, self.t)                  # energy of the impulse response

    def indices(self, tf):
        _, yr = signal.step(tf, T=self.t)
        e = self.y - yr
        ise = _trapz(e ** 2, self.t)
        return {"ISE":  PAPER_SCALE * ise,
                "RISE": ise / self.h2,                    # relative ISE (not scaled)
                "IAE":  PAPER_SCALE * _trapz(np.abs(e), self.t),
                "ITAE": PAPER_SCALE * _trapz(self.t * np.abs(e), self.t)}


# =====================================================================
#                                MAIN
# =====================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(RANDOM_SEED)
    plant = (np.array(PLANT_NUM, float), np.array(PLANT_DEN, float))
    fmt_tf(*plant, f"Original plant G(s)  (order {len(PLANT_DEN) - 1})")

    # ---------- data generation & hold-out split ----------
    u = make_random_input(N_SAMPLES, rng, INPUT_MIN, INPUT_MAX, HOLD_MIN, HOLD_MAX)
    y = simulate_tf(*plant, u, TS)
    n_tr, n_va = int(TRAIN_FRAC * N_SAMPLES), int(VAL_FRAC * N_SAMPLES)
    u_tr, y_tr = u[:n_tr], y[:n_tr]
    u_va, y_va = u[n_tr:n_tr + n_va], y[n_tr:n_tr + n_va]
    u_te, y_te = u[n_tr + n_va:], y[n_tr + n_va:]
    if CUSTOM_TEST_INPUT is not None:
        t_c = np.arange(len(u_te)) * TS
        u_te = CUSTOM_TEST_INPUT(t_c) if callable(CUSTOM_TEST_INPUT) else np.asarray(CUSTOM_TEST_INPUT, float)
        y_te = simulate_tf(*plant, u_te, TS)
    t_te = np.arange(len(u_te)) * TS

    # ---------- JRNN training ----------
    print("\nTraining JRNN (Levenberg-Marquardt) ...")
    jrnn = JRNN(HIDDEN_NEURONS, seed=RANDOM_SEED)
    val_mse = jrnn.train(u_tr, y_tr, u_va, y_va, JRNN_MAX_ITERS, free_run_stage=JRNN_FREE_RUN_FINETUNE)
    y_nn_one = jrnn.one_step(u_te, y_te)
    y_nn_free = jrnn.free_run(u_te) if JRNN_RESPONSE == "free_run" else None
    print(f"  validation MSE                 : {val_mse:.3e}")
    print(f"  test MSE JRNN one-step         : {np.mean((y_nn_one - y_te) ** 2):.3e}")
    if JRNN_RESPONSE == "free_run":
        print(f"  test MSE JRNN free-run         : {np.mean((y_nn_free - y_te) ** 2):.3e}")
    y_nn_test = y_nn_free if JRNN_RESPONSE == "free_run" else y_nn_one

    # ---------- IV estimation on the JRNN response ----------
    # identification signal: a fresh random input passed through the trained JRNN
    u_id = make_random_input(3000, np.random.default_rng(RANDOM_SEED + 1),
                             INPUT_MIN, INPUT_MAX, HOLD_MIN, HOLD_MAX)
    if JRNN_RESPONSE == "free_run":
        y_id = jrnn.free_run(u_id)
    else:
        y_id = jrnn.one_step(u_id, simulate_tf(*plant, u_id, TS))

    G_ivjrnn_c = iv_estimate(u_id, y_id, TS, IV_ORDER_CASCADE, iters=IV_ITERATIONS, p0=IV_PREFILTER_P)
    G_ivjrnn_f = iv_estimate(u_id, y_id, TS, FAIR_ORDER, iters=IV_ITERATIONS, p0=IV_PREFILTER_P)

    # ---------- Balanced truncation ----------
    G_bt_c, hsv = balanced_reduction(*plant, BT_ORDER_CASCADE, BT_METHOD)
    G_bt_f, _ = balanced_reduction(*plant, FAIR_ORDER, BT_METHOD)
    print("\nHankel singular values:", np.round(hsv, 5))

    # ---------- Cascade ----------
    G_btjrnn = series(G_bt_c, G_ivjrnn_c)                    # eq. (2)

    fmt_tf(*G_bt_c,      f"G_BT  (order {BT_ORDER_CASCADE}, used in cascade)   [paper eq. 4]")
    fmt_tf(*G_ivjrnn_c,  f"G_IV-JRNN (order {IV_ORDER_CASCADE}, used in cascade) [paper eq. 5]")
    fmt_tf(*G_btjrnn,    f"G_BT-JRNN = G_BT * G_IV-JRNN (order {len(G_btjrnn[1]) - 1}) [paper eq. 6]")
    fmt_tf(*G_bt_f,      f"G_BT alone (order {FAIR_ORDER})")
    fmt_tf(*G_ivjrnn_f,  f"G_JRNN alone, IV-estimated (order {FAIR_ORDER})")

    # ---------- Results table ----------
    w = np.logspace(np.log10(W_MIN), np.log10(W_MAX), N_FREQ)
    models = {
        f"JRNN  (IV est., order {FAIR_ORDER})": G_ivjrnn_f,
        f"BT    (order {FAIR_ORDER})":           G_bt_f,
        f"BT-JRNN (order {len(G_btjrnn[1]) - 1})": G_btjrnn,
    }
    rows = []
    for name, tf in models.items():
        m_mse, p_mse = bode_mse(plant, tf, w)
        t_mse = np.mean((simulate_tf(*tf, u_te, TS) - y_te) ** 2)
        rows.append((name, m_mse, p_mse, t_mse))

    print("\n" + "=" * 86)
    print("TABLE I  -  MSE between ORIGINAL plant and each reduced model")
    print("=" * 86)
    print(f"{'Technique':<26}{'Magnitude_MSE':>15}{'Phase_MSE':>14}{'Time_MSE (test)':>18}{'Freq range':>16}")
    print("-" * 86)
    for name, m, p, t in rows:
        print(f"{name:<26}{m:>15.4g}{p:>14.4g}{t:>18.4g}{f'{W_MIN:g}-{W_MAX:g} rad/s':>16}")
    print("=" * 86)
    unit = "dB^2" if MAG_UNITS == "dB" else "linear |G|^2"
    print(f"Magnitude_MSE in {unit}, Phase_MSE in deg^2, Time_MSE in (output units)^2.")

    # ---------- Table as PNG ----------
    unit = "dB\u00b2" if MAG_UNITS == "dB" else "linear"
    col_labels = ["Technique", f"Magnitude_MSE\n({unit})", "Phase_MSE\n(deg\u00b2)",
                  "Time_MSE (test)", "Frequency range"]
    cell_text = [[name, f"{m:.4g}", f"{p:.4g}", f"{t:.4g}", f"{W_MIN:g}-{W_MAX:g} (rad/s)"]
                 for name, m, p, t in rows]
    fig, ax = plt.subplots(figsize=(11, 1.4 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.0)
    tbl.auto_set_column_width(range(len(col_labels)))
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2f4b7c"); cell.set_text_props(color="white", weight="bold")
        elif r == len(rows):                      # highlight BT-JRNN row
            cell.set_facecolor("#e8eef9")
    ax.set_title("TABLE I - MSE between the original plant and each reduced model",
                 fontsize=12, weight="bold", pad=10)
    fig.tight_layout()
    if SAVE_FIGS: fig.savefig("table_mse.png", dpi=200, bbox_inches="tight")

    # ---------- Step-response error indices (ISE, RISE, IAE, ITAE) ----------
    ref = StepReference(plant)
    IDX = ["ISE", "RISE", "IAE", "ITAE"]
    err_rows = [(name, ref.indices(tf)) for name, tf in models.items()]

    print("\n" + "=" * 86)
    print("TABLE II -  Step-response error indices (original plant vs. each reduced model)")
    print("=" * 86)
    print(f"{'Technique':<26}" + "".join(f"{k:>15}" for k in IDX))
    print("-" * 86)
    for name, d in err_rows:
        print(f"{name:<26}" + "".join(f"{d[k]:>15.4g}" for k in IDX))
    print("=" * 86)
    print(f"ISE, IAE, ITAE are multiplied by PAPER_SCALE = {PAPER_SCALE:g}; RISE = ISE / (impulse-response energy).")

    # ---------- Error-index table as PNG (separate from the MSE table) ----------
    best = {k: min(d[k] for _, d in err_rows) for k in IDX}
    fig, ax = plt.subplots(figsize=(12, 1.6 + 0.55 * len(err_rows)))
    ax.axis("off")
    cell_text = [[name] + [f"{d[k]:.4g}" for k in IDX] for name, d in err_rows]
    tbl = ax.table(cellText=cell_text, colLabels=["Technique"] + IDX, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 2.2)
    tbl.auto_set_column_width(range(len(IDX) + 1))
    for (r_, c_), cell in tbl.get_celld().items():
        if r_ == 0:
            cell.set_facecolor("#1b6b4a"); cell.set_text_props(color="white", weight="bold")
        elif c_ >= 1 and cell_text[r_ - 1][c_] == f"{best[IDX[c_ - 1]]:.4g}":
            cell.set_text_props(color="darkgreen", weight="bold")     # best value of each index
            cell.set_facecolor("#d9f2d9")
    ax.set_title("TABLE II - Step-response error indices (best value of each index in bold green)",
                 fontsize=12, weight="bold", pad=10)
    fig.tight_layout()
    if SAVE_FIGS: fig.savefig("table_step_errors.png", dpi=200, bbox_inches="tight")

    # ---------- Figures ----------
    # Fig 1: JRNN testing (like paper Fig. 4)
    k = slice(0, min(500, len(u_te)))
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    ax[0, 0].plot(u_te[k]);           ax[0, 0].set_title("Input")
    ax[0, 1].plot(y_te[k]);           ax[0, 1].set_title("Plant output")
    ax[1, 0].plot((y_te - y_nn_test)[k], lw=0.8); ax[1, 0].set_title(f"Error (JRNN, {JRNN_RESPONSE})")
    ax[1, 1].plot(y_nn_test[k]);      ax[1, 1].set_title("NN output")
    for a in ax.ravel(): a.set_xlabel("Samples"); a.set_ylabel("Amplitude"); a.grid(alpha=.3)
    fig.suptitle("Testing the JRNN model"); fig.tight_layout()
    if SAVE_FIGS: fig.savefig("fig_jrnn_test.png", dpi=140)

    # Fig 2: Bode comparison
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    mr, pr = bode_data(*plant, w)
    a1.semilogx(w, mr, "k", lw=2, label="Original (6th)")
    a2.semilogx(w, pr, "k", lw=2, label="Original (6th)")
    for (name, tf), ls in zip(models.items(), ["--", "-.", "-"]):
        mm, pm = bode_data(*tf, w)
        pm = pm - 360 * np.round((pm[0] - pr[0]) / 360)
        a1.semilogx(w, mm, ls, label=name); a2.semilogx(w, pm, ls, label=name)
    a1.set_ylabel("Magnitude (dB)" if MAG_UNITS == "dB" else "Magnitude |G|"); a2.set_ylabel("Phase (deg)"); a2.set_xlabel("Frequency (rad/s)")
    a1.legend(); a1.grid(which="both", alpha=.3); a2.grid(which="both", alpha=.3)
    fig.suptitle("Bode plot comparison"); fig.tight_layout()
    if SAVE_FIGS: fig.savefig("fig_bode.png", dpi=140)

    # Fig 3: step response + test-signal response
    fig, (b1, b2) = plt.subplots(2, 1, figsize=(9, 7))
    tt = np.arange(0, 8, TS); step = np.ones_like(tt)
    b1.plot(tt, simulate_tf(*plant, step, TS), "k", lw=2, label="Original (6th)")
    b2.plot(t_te[k], y_te[k], "k", lw=2, label="Original (6th)")
    for (name, tf), ls in zip(models.items(), ["--", "-.", "-"]):
        b1.plot(tt, simulate_tf(*tf, step, TS), ls, label=name)
        b2.plot(t_te[k], simulate_tf(*tf, u_te, TS)[k], ls, label=name)
    b1.set_title("Step response"); b2.set_title("Response to the test input")
    for a in (b1, b2): a.set_xlabel("Time (s)"); a.grid(alpha=.3); a.legend()
    fig.tight_layout()
    if SAVE_FIGS: fig.savefig("fig_step.png", dpi=140)
    plt.show()
