"""M/D/1 queueing model for signalised intersections."""

import numpy as np
import matplotlib.pyplot as plt


def md1_queue_length(rho):
    """M/D/1 expected queue length: Lq = rho^2 / (2*(1-rho))."""
    return rho ** 2 / (2 * (1 - rho))


def md1_waiting_time(rho, mu):
    """M/D/1 expected waiting time: Wq = rho / (2*mu*(1-rho))."""
    return rho / (2 * mu * (1 - rho))


def simulate(lambda_range=None, mu=0.5, n_intersections=4):
    """Simulate M/D/1 queueing for a signalised corridor.

    Args:
        lambda_range: array of arrival rates to sweep
        mu: service rate (vehicles per second during green)
        n_intersections: number of intersections in corridor

    Returns dict with utilisation, queue lengths, waiting times.
    """
    if lambda_range is None:
        lambda_range = np.linspace(0.05, 0.48, 100)

    rho_values = lambda_range / mu
    queue_lengths = md1_queue_length(rho_values)
    waiting_times = md1_waiting_time(rho_values, mu)

    # Tandem queue approximation (Jackson network-like)
    # Total delay through corridor ~ sum of individual delays
    corridor_delay = n_intersections * waiting_times
    corridor_queue = n_intersections * queue_lengths

    return {"lambda": lambda_range, "mu": mu,
            "rho": rho_values,
            "queue_length": queue_lengths,
            "waiting_time": waiting_times,
            "corridor_delay": corridor_delay,
            "corridor_queue": corridor_queue,
            "n_intersections": n_intersections}


def plot(results, save_path="results/queueing_md1.png"):
    """Plot queue length and delay vs utilisation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    rho = results["rho"]

    # Single intersection queue length
    ax1.plot(rho, results["queue_length"], "b-", linewidth=2)
    ax1.set_xlabel("Utilisation ρ = λ/μ")
    ax1.set_ylabel("Expected Queue Length (Lq)")
    ax1.set_title("M/D/1: Queue Length vs Utilisation")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, min(50, results["queue_length"][-1] * 1.1))
    ax1.axvline(x=0.85, color="r", linestyle="--", alpha=0.5, label="ρ=0.85")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Single intersection waiting time
    ax2.plot(rho, results["waiting_time"], "r-", linewidth=2)
    ax2.set_xlabel("Utilisation ρ")
    ax2.set_ylabel("Expected Waiting Time (s)")
    ax2.set_title("M/D/1: Waiting Time vs Utilisation")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, min(100, results["waiting_time"][-1] * 1.1))
    ax2.grid(True, alpha=0.3)

    # Corridor total queue
    ax3.plot(rho, results["corridor_queue"], "g-", linewidth=2)
    ax3.set_xlabel("Utilisation ρ")
    ax3.set_ylabel("Total Queue Length")
    ax3.set_title(f"Corridor ({results['n_intersections']} intersections): Total Queue")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, min(200, results["corridor_queue"][-1] * 1.1))
    ax3.grid(True, alpha=0.3)

    # Corridor total delay
    ax4.plot(rho, results["corridor_delay"], "m-", linewidth=2)
    ax4.set_xlabel("Utilisation ρ")
    ax4.set_ylabel("Total Corridor Delay (s)")
    ax4.set_title("Corridor: Total Delay vs Utilisation")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, min(400, results["corridor_delay"][-1] * 1.1))
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
