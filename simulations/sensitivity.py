"""Parameter sensitivity analysis for traffic models."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def sweep_idm_headway():
    """Sweep IDM safe time headway T and measure jam severity."""
    from trafficjams import idm
    headways = np.linspace(0.5, 3.0, 12)
    speed_stds = []
    min_speeds = []

    for T in headways:
        r = idm.simulate(n_vehicles=50, T=80, T_headway=T)
        steady = r["velocities"][len(r["times"]) // 2:]
        speed_stds.append(steady.std())
        min_speeds.append(steady.min())

    return headways, speed_stds, min_speeds


def sweep_nasch_slowdown():
    """Sweep NaSch randomisation probability."""
    from trafficjams import nagel_schreckenberg
    probs = np.linspace(0.0, 0.7, 12)
    avg_speeds = []
    flow_rates = []

    for p in probs:
        r = nagel_schreckenberg.simulate(road_length=500, n_vehicles=100, T=200, p_slow=p)
        st = r["spacetime"][100:]
        occupied = st >= 0
        speed_vals = st.copy().astype(float)
        speed_vals[~occupied] = 0
        avg_v = speed_vals.sum() / max(occupied.sum(), 1)
        avg_speeds.append(avg_v)
        flow_rates.append(100 / 500 * avg_v)

    return probs, avg_speeds, flow_rates


def sweep_bando_kappa():
    """Sweep Bando sensitivity parameter kappa."""
    from trafficjams import bando
    kappas = np.linspace(0.3, 3.0, 12)
    speed_stds = []
    convergence_times = []

    for k in kappas:
        r = bando.simulate(n_vehicles=40, T=150, kappa=k)
        vels = r["velocities"]
        speed_stds.append(vels[-50:].std())
        # Estimate convergence: when std stabilises
        stds = [vels[i:i+10].std() for i in range(0, len(vels)-10, 10)]
        convergence_times.append(len(stds))

    return kappas, speed_stds, convergence_times


def sweep_queueing_service():
    """Sweep service rate in queueing model."""
    from trafficjams import queueing
    mus = np.linspace(0.3, 1.0, 12)
    lam = 0.4  # fixed arrival rate
    queue_lengths = []
    delays = []

    for mu in mus:
        rho = lam / mu
        if rho < 1:
            queue_lengths.append(rho**2 / (2 * (1 - rho)))
            delays.append(rho / (2 * mu * (1 - rho)))
        else:
            queue_lengths.append(np.inf)
            delays.append(np.inf)

    return mus, queue_lengths, delays


def plot_all(save_path=None):
    save_path = save_path or os.path.join(RESULTS_DIR, "sensitivity.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # IDM headway sweep
    print("Sweeping IDM headway...")
    T_vals, stds, mins = sweep_idm_headway()
    ax = axes[0, 0]
    ax.plot(T_vals, stds, "b-o", label="Speed std dev")
    ax.set_xlabel("Safe Time Headway T (s)")
    ax.set_ylabel("Speed Variability (m/s)")
    ax.set_title("IDM: Jam Severity vs Headway")
    ax2 = ax.twinx()
    ax2.plot(T_vals, mins, "r--s", label="Min speed")
    ax2.set_ylabel("Min Speed (m/s)", color="red")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # NaSch slowdown sweep
    print("Sweeping NaSch slowdown probability...")
    probs, speeds, flows = sweep_nasch_slowdown()
    ax = axes[0, 1]
    ax.plot(probs, speeds, "g-o", label="Avg speed")
    ax.set_xlabel("Slowdown Probability p")
    ax.set_ylabel("Average Speed")
    ax.set_title("NaSch: Speed vs Randomisation")
    ax2 = ax.twinx()
    ax2.plot(probs, flows, "m--s", label="Flow")
    ax2.set_ylabel("Flow Rate", color="purple")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Bando kappa sweep
    print("Sweeping Bando kappa...")
    kappas, bstds, _ = sweep_bando_kappa()
    ax = axes[1, 0]
    ax.plot(kappas, bstds, "r-o")
    ax.set_xlabel("Sensitivity κ")
    ax.set_ylabel("Steady-State Speed Std Dev")
    ax.set_title("Bando OVM: Instability vs Sensitivity")
    ax.axvspan(0.5, 1.5, alpha=0.1, color="red", label="Unstable region")
    ax.legend()

    # Queueing service sweep
    print("Sweeping queueing service rate...")
    mus, ql, dl = sweep_queueing_service()
    ax = axes[1, 1]
    ql_clipped = [min(q, 50) for q in ql]
    dl_clipped = [min(d, 100) for d in dl]
    ax.plot(mus, ql_clipped, "b-o", label="Queue length")
    ax.set_xlabel("Service Rate μ")
    ax.set_ylabel("Queue Length")
    ax.set_title("M/D/1: Queue vs Service Rate (λ=0.4)")
    ax2 = ax.twinx()
    ax2.plot(mus, dl_clipped, "r--s", label="Delay")
    ax2.set_ylabel("Waiting Time (s)", color="red")
    ax.axvline(x=0.4, color="gray", linestyle=":", label="ρ=1 boundary")
    ax.legend(loc="upper right")
    ax2.legend(loc="center right")

    plt.suptitle("Parameter Sensitivity Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_all()
