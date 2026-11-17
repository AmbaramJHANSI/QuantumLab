"""Quantum-kernel SVM on the Iris dataset.

Idea: encode each 4-feature flower into a 4-qubit state with a ZZ-style feature map.
The kernel K(x, y) = |<phi(x)|phi(y)>|^2 is computed from the quantum states and handed to a
classical SVM. Compared against a classical RBF-SVM baseline.

Protocol (no test-set tuning): for each random 70/30 split, hyperparameters of BOTH models are chosen
by 5-fold cross-validation on the training part only; the chosen model is then scored once on the test part.

Run:  python quantum_iris.py                 (uses Qiskit if installed, else NumPy simulator)
      python quantum_iris.py --backend numpy
      python quantum_iris.py --reps 2 --scale 1.0   (fix the feature map instead of tuning it)
"""
import argparse
import itertools

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

N_QUBITS = 4
PAIRS = list(itertools.combinations(range(N_QUBITS), 2))


# ---------- Backend 1: Qiskit ----------
def feature_map_circuit(x, reps=2):
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_QUBITS)
    for _ in range(reps):
        qc.h(range(N_QUBITS))
        for i in range(N_QUBITS):
            qc.p(2 * x[i], i)
        for i, j in PAIRS:
            qc.cx(i, j)
            qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
            qc.cx(i, j)
    return qc


def states_qiskit(X, reps):
    from qiskit.quantum_info import Statevector

    return np.array([Statevector(feature_map_circuit(x, reps)).data for x in X])


# ---------- Backend 2: NumPy statevector simulator (same circuit) ----------
def _apply_1q(state, gate, q):
    state = np.moveaxis(state.reshape([2] * N_QUBITS), q, 0)
    state = np.tensordot(gate, state, axes=(1, 0))
    return np.moveaxis(state, 0, q).reshape(-1)


def _apply_cx(state, c, t):
    s = state.reshape([2] * N_QUBITS).copy()
    idx = [slice(None)] * N_QUBITS
    idx[c] = 1
    sub = s[tuple(idx)]
    t_axis = t if t < c else t - 1
    s[tuple(idx)] = np.flip(sub, axis=t_axis)
    return s.reshape(-1)


def state_numpy(x, reps):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    state = np.zeros(2 ** N_QUBITS, dtype=complex)
    state[0] = 1
    P = lambda th: np.diag([1, np.exp(1j * th)])
    for _ in range(reps):
        for q in range(N_QUBITS):
            state = _apply_1q(state, H, q)
        for q in range(N_QUBITS):
            state = _apply_1q(state, P(2 * x[q]), q)
        for i, j in PAIRS:
            state = _apply_cx(state, i, j)
            state = _apply_1q(state, P(2 * (np.pi - x[i]) * (np.pi - x[j])), j)
            state = _apply_cx(state, i, j)
    return state


def to_qiskit_order(state):
    """My NumPy simulator puts qubit 0 first (most significant); Qiskit puts qubit 0 last. Reverse the bit order."""
    return state.reshape([2] * N_QUBITS).transpose(tuple(reversed(range(N_QUBITS)))).reshape(-1)


def states_numpy(X, reps):
    return np.array([state_numpy(x, reps) for x in X])


# ---------- Kernel + experiment ----------
def get_states(X, reps, backend):
    return states_qiskit(X, reps) if backend == "qiskit" else states_numpy(X, reps)


def kernel_from_states(A, B):
    return np.abs(A.conj() @ B.T) ** 2


def tune_quantum(S_by_cfg, ytr, Cs, folds=5, seed=0):
    """5-fold CV over (reps, scale, C). S_by_cfg[(reps, scale)] = (train_states, test_states)."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    best = (-1, None)
    for cfg, (Str, _) in S_by_cfg.items():
        K = kernel_from_states(Str, Str)
        for C in Cs:
            accs = []
            for tr, va in skf.split(Str, ytr):
                m = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], ytr[tr])
                accs.append(accuracy_score(ytr[va], m.predict(K[np.ix_(va, tr)])))
            if np.mean(accs) > best[0]:
                best = (np.mean(accs), (cfg, C))
    return best


def tune_classical(Xtr, ytr, Cs, gammas, folds=5, seed=0):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    best = (-1, None)
    for C in Cs:
        for g in gammas:
            accs = []
            for tr, va in skf.split(Xtr, ytr):
                m = SVC(kernel="rbf", C=C, gamma=g).fit(Xtr[tr], ytr[tr])
                accs.append(accuracy_score(ytr[va], m.predict(Xtr[va])))
            if np.mean(accs) > best[0]:
                best = (np.mean(accs), (C, g))
    return best


def run_once(seed, backend, reps_grid, scale_grid, Cs, gammas):
    data = load_iris()
    Xtr, Xte, ytr, yte = train_test_split(
        data.data, data.target, test_size=0.3, random_state=seed, stratify=data.target
    )
    unit = MinMaxScaler((0, 1)).fit(Xtr)          # scaler fitted on training data only
    Utr, Ute = unit.transform(Xtr), unit.transform(Xte)

    # ---- quantum: tune on train via CV, then test once ----
    S = {}
    for reps in reps_grid:
        for scale in scale_grid:
            S[(reps, scale)] = (get_states(np.pi * scale * Utr, reps, backend),
                                get_states(np.pi * scale * Ute, reps, backend))
    q_cv, ((reps, scale), C) = tune_quantum(S, ytr, Cs, seed=seed)
    Str, Ste = S[(reps, scale)]
    Ktr, Kte = kernel_from_states(Str, Str), kernel_from_states(Ste, Str)
    q_pred = SVC(kernel="precomputed", C=C).fit(Ktr, ytr).predict(Kte)

    # ---- classical baseline: same protocol ----
    c_cv, (Cc, g) = tune_classical(Utr, ytr, Cs, gammas, seed=seed)
    c_pred = SVC(kernel="rbf", C=Cc, gamma=g).fit(Utr, ytr).predict(Ute)

    info = dict(q_cfg=(reps, scale, C), q_cv=q_cv, c_cfg=(Cc, g), c_cv=c_cv)
    return yte, q_pred, c_pred, Ktr, ytr, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["auto", "qiskit", "numpy"], default="auto")
    ap.add_argument("--reps", type=int, default=None, help="fix feature-map repetitions (default: tune over 1,2)")
    ap.add_argument("--scale", type=float, default=None, help="fix feature range [0, pi*scale] (default: tune)")
    ap.add_argument("--seeds", type=int, default=5, help="number of random train/test splits")
    a = ap.parse_args()

    backend = a.backend
    if backend == "auto":
        try:
            import qiskit  # noqa: F401
            backend = "qiskit"
        except ImportError:
            backend = "numpy"
    reps_grid = [a.reps] if a.reps else [1, 2]
    scale_grid = [a.scale] if a.scale else [0.05, 0.1, 0.25, 0.5, 1.0]
    Cs, gammas = [1, 10, 100], [0.5, 1, 2, 5, 10, 20]
    print(f"backend={backend}, qubits={N_QUBITS}, reps grid={reps_grid}, scale grid={scale_grid}")

    # Self-test: both simulators must agree if Qiskit is available
    if backend == "qiskit":
        x = np.array([0.3, 1.1, 2.0, 2.9])
        f = abs(np.vdot(states_qiskit([x], reps_grid[0])[0], to_qiskit_order(state_numpy(x, reps_grid[0])))) ** 2
        print(f"self-test: Qiskit vs NumPy simulator fidelity = {f:.6f} (should be 1.0)")

    q_acc, c_acc = [], []
    for s in range(a.seeds):
        yte, qp, cp, Ktr, ytr, info = run_once(s, backend, reps_grid, scale_grid, Cs, gammas)
        q_acc.append(accuracy_score(yte, qp))
        c_acc.append(accuracy_score(yte, cp))
        print(f"seed {s}: quantum {q_acc[-1]:.3f} (CV {info['q_cv']:.3f}, reps/scale/C={info['q_cfg']}) | "
              f"classical {c_acc[-1]:.3f} (CV {info['c_cv']:.3f}, C/gamma={info['c_cfg']})")
    print(f"\nTest accuracy over {a.seeds} splits (hyperparameters chosen by CV on training data only)")
    print(f"  Quantum kernel SVM : {np.mean(q_acc):.4f} +/- {np.std(q_acc):.4f}")
    print(f"  Classical RBF SVM  : {np.mean(c_acc):.4f} +/- {np.std(c_acc):.4f}")
    print("Confusion matrix (quantum, last split):\n", confusion_matrix(yte, qp))

    try:
        import matplotlib.pyplot as plt

        order = np.argsort(ytr, kind="stable")           # sort training samples by class
        Ks = Ktr[np.ix_(order, order)]
        bounds = np.cumsum(np.bincount(ytr))[:-1] - 0.5
        names = load_iris().target_names

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        im = ax[0].imshow(Ks, cmap="viridis", vmin=0, vmax=1)
        for b in bounds:
            ax[0].axhline(b, color="white", lw=1); ax[0].axvline(b, color="white", lw=1)
        mids = (np.r_[-0.5, bounds] + np.r_[bounds, len(ytr) - 0.5]) / 2
        ax[0].set_xticks(mids); ax[0].set_xticklabels(names)
        ax[0].set_yticks(mids); ax[0].set_yticklabels(names)
        ax[0].set_title("Quantum kernel (train, sorted by class)")
        plt.colorbar(im, ax=ax[0])
        ax[1].bar(["Quantum kernel", "Classical RBF"], [np.mean(q_acc), np.mean(c_acc)],
                  yerr=[np.std(q_acc), np.std(c_acc)], capsize=6)
        ax[1].set_ylim(0.5, 1.0)
        ax[1].set_ylabel(f"Mean test accuracy ({a.seeds} splits)")
        plt.tight_layout()
        plt.savefig("quantum_iris_results.png", dpi=150)
        print("saved quantum_iris_results.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
