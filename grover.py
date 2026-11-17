"""Grover's search on a simulator, with a noise study.

Run:  python grover.py            (default: 4 qubits, target '1011')
      python grover.py --n 5 --target 10110
"""
import argparse
import math

import numpy as np


# ---------- pure-math part (no Qiskit needed) ----------
def optimal_iterations(n: int) -> int:
    """Grover iterations that maximise success for one marked item (exact: round(pi/(4*theta) - 1/2))."""
    theta = math.asin(1 / math.sqrt(2 ** n))
    return max(1, round(math.pi / (4 * theta) - 0.5))


def theory_success(n: int, k: int) -> float:
    """Ideal probability of measuring the marked state after k iterations."""
    theta = math.asin(1 / math.sqrt(2 ** n))
    return math.sin((2 * k + 1) * theta) ** 2


# ---------- Qiskit part ----------
def build_grover(n: int, target: str, iterations: int):
    from qiskit import QuantumCircuit

    assert len(target) == n and set(target) <= {"0", "1"}
    qc = QuantumCircuit(n)
    qc.h(range(n))

    def mcz():  # multi-controlled Z on all qubits
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)

    for _ in range(iterations):
        # Oracle: phase-flip |target>. Qiskit is little-endian: qubit i <-> target[n-1-i]
        zeros = [i for i in range(n) if target[n - 1 - i] == "0"]
        if zeros:
            qc.x(zeros)
        mcz()
        if zeros:
            qc.x(zeros)
        # Diffuser: inversion about the mean
        qc.h(range(n))
        qc.x(range(n))
        mcz()
        qc.x(range(n))
        qc.h(range(n))
    qc.measure_all()
    return qc


def run(qc, shots=4096, error_1q=0.0, error_2q=0.0):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    basis = ["u", "cx"]
    noise = None
    if error_1q or error_2q:
        noise = NoiseModel(basis_gates=basis)
        noise.add_all_qubit_quantum_error(depolarizing_error(error_1q, 1), ["u"])
        noise.add_all_qubit_quantum_error(depolarizing_error(error_2q, 2), ["cx"])
    sim = AerSimulator(noise_model=noise)
    tqc = transpile(qc, basis_gates=basis, optimization_level=1)  # no backend= here: avoids the ambiguity warning
    counts = sim.run(tqc, shots=shots).result().get_counts()
    return counts, tqc.size()


def success_rate(counts, target):
    total = sum(counts.values())
    return counts.get(target, 0) / total


def main():
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--target", default="1011")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--skip-scaling", action="store_true", help="skip the n=3..8 scaling study")
    a = ap.parse_args()
    n, target = a.n, a.target

    best = optimal_iterations(n)
    print(f"n={n}, N={2**n}, target={target}, optimal iterations={best}")

    # 1) success vs iterations: simulation vs theory
    ks = list(range(0, 2 * best + 2))
    sim_p = []
    for k in ks:
        counts, _ = run(build_grover(n, target, k), a.shots)
        sim_p.append(success_rate(counts, target))
    th_p = [theory_success(n, k) for k in ks]

    # 2) effect of gate noise at the optimal iteration count
    errs = [0.0, 0.001, 0.003, 0.01, 0.02]
    noisy_p = []
    for e in errs:
        counts, size = run(build_grover(n, target, best), a.shots, e, 10 * e)
        noisy_p.append(success_rate(counts, target))
        print(f"  gate error 1q={e:.3f}, 2q={10*e:.3f} -> success {noisy_p[-1]:.3f} (circuit size {size})")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks, th_p, "o-", label="theory")
    ax[0].plot(ks, sim_p, "x--", label="Aer simulation")
    ax[0].axvline(best, color="gray", ls=":")
    ax[0].set(xlabel="Grover iterations", ylabel="P(target)", title=f"Grover, n={n}")
    ax[0].legend()
    ax[1].plot(errs, noisy_p, "o-")
    ax[1].axhline(1 / 2**n, color="red", ls=":", label="random guess")
    ax[1].set(xlabel="1-qubit depolarizing error", ylabel="P(target)", title="Effect of noise")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("grover_results.png", dpi=150)
    print("saved grover_results.png")

    if not a.skip_scaling:
        scaling_study(a.shots)


def scaling_study(shots=2048, n_range=range(3, 9)):
    """How does the optimal iteration count grow with search-space size N = 2^n?

    For each n: simulate k-1, k, k+1 iterations (k = formula value) and record which k is best.
    Expectation: optimal k ~ (pi/4) * sqrt(N), i.e. quadratic speed-up over ~N/2 classical guesses.
    """
    import matplotlib.pyplot as plt

    rows = []
    print("\nScaling study (target = all ones):")
    print(" n |    N | sqrtN | formula k | best simulated k | P(sim) | P(theory)")
    for n in n_range:
        k0 = optimal_iterations(n)
        target = "1" * n
        cands = [k for k in (k0 - 1, k0, k0 + 1) if k >= 1]
        probs = {}
        for k in cands:
            counts, _ = run(build_grover(n, target, k), shots)
            probs[k] = success_rate(counts, target)
        kb = max(probs, key=probs.get)
        rows.append((n, 2 ** n, math.sqrt(2 ** n), k0, kb, probs[kb], theory_success(n, kb)))
        r = rows[-1]
        print(f" {r[0]} | {r[1]:4d} | {r[2]:5.2f} | {r[3]:9d} | {r[4]:16d} | {r[5]:.3f}  | {r[6]:.3f}")

    sq = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.linspace(sq.min() * 0.9, sq.max() * 1.05, 50)
    ax[0].plot(xs, np.pi / 4 * xs, "k:", label=r"$(\pi/4)\sqrt{N}$")
    ax[0].plot(sq, [r[3] for r in rows], "o", ms=9, mfc="none", label="formula (rounded)")
    ax[0].plot(sq, [r[4] for r in rows], "x", ms=8, label="best k in simulation")
    for r in rows:
        ax[0].annotate(f"n={r[0]}", (r[2], r[4]), textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax[0].set(xlabel=r"$\sqrt{N}$, $N=2^n$", ylabel="Optimal Grover iterations",
              title="Iterations grow as sqrt(N)")
    ax[0].legend()

    ns = [r[0] for r in rows]
    ax[1].plot(ns, [r[6] for r in rows], "o-", label="theory")
    ax[1].plot(ns, [r[5] for r in rows], "x--", label="Aer simulation")
    ax[1].set(xlabel="Number of qubits n", ylabel="P(target) at best k", ylim=(0, 1.05),
              title="Success probability vs problem size")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("grover_scaling.png", dpi=150)
    print("saved grover_scaling.png")


if __name__ == "__main__":
    main()
