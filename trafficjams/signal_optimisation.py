"""Signal optimisation: optimise green phase splits for a signalised corridor."""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def corridor_delay(green_splits, arrivals, cycle_time=90, saturation_flow=0.5):
    """Total delay for a corridor of signalised intersections.

    Args:
        green_splits: array of green times for each intersection
        arrivals: array of arrival rates (veh/s) at each intersection
        cycle_time: total cycle time (s)
        saturation_flow: saturation flow rate during green (veh/s)

    Returns total delay across all intersections.
    """
    total_delay = 0
    n = len(green_splits)
    for i in range(n):
        g = green_splits[i]
        lam = arrivals[i]
        mu = saturation_flow * g / cycle_time  # effective service rate
        if mu <= 0 or mu <= lam:
            total_delay += 1e6  # infeasible
            continue
        rho = lam / mu
        if rho >= 1:
            total_delay += 1e6
            continue
        # M/D/1 delay formula
        Wq = rho / (2 * mu * (1 - rho))
        # Webster's uniform delay component
        red = cycle_time - g
        d_uniform = red**2 / (2 * cycle_time)
        total_delay += lam * (Wq + d_uniform)
    return total_delay


def simulate(n_intersections=5, cycle_time=90):
    """Optimise green splits for a corridor.

    Returns dict with optimal splits, delays for optimal vs uniform.
    """
    # Arrival rates (varying by intersection — typical for Union St corridor)
    # Asymmetric demand creates scope for optimisation
    arrivals = np.array([0.08, 0.15, 0.22, 0.17, 0.06])[:n_intersections]

    # Uniform splits (baseline — equal green for all)
    min_green = 20
    max_green = cycle_time - 15  # leave time for other phases + amber
    uniform_green = np.ones(n_intersections) * 50  # 50s green each
    uniform_delay = corridor_delay(uniform_green, arrivals, cycle_time)

    # Optimise: minimise total delay subject to total green budget
    # (models shared infrastructure / coordinated signal plan)
    total_green_budget = n_intersections * 45  # average 45s per intersection

    def objective(g):
        return corridor_delay(g, arrivals, cycle_time)

    bounds = [(min_green, max_green)] * n_intersections
    constraints = [{"type": "eq",
                    "fun": lambda g: np.sum(g) - total_green_budget}]
    result = minimize(objective, uniform_green, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    optimal_green = result.x
    optimal_delay = result.fun

    # Sweep for each intersection to show sensitivity
    sweep_results = []
    for i in range(n_intersections):
        greens = np.linspace(min_green, max_green, 50)
        delays = []
        for g in greens:
            splits = optimal_green.copy()
            splits[i] = g
            delays.append(corridor_delay(splits, arrivals, cycle_time))
        sweep_results.append((greens, delays))

    return {
        "n_intersections": n_intersections,
        "cycle_time": cycle_time,
        "arrivals": arrivals,
        "uniform_green": uniform_green,
        "uniform_delay": uniform_delay,
        "optimal_green": optimal_green,
        "optimal_delay": optimal_delay,
        "sweep_results": sweep_results,
        "improvement_pct": (1 - optimal_delay / uniform_delay) * 100,
    }


def plot(results, save_path="results/signal_optimisation.png"):
    """Plot signal optimisation results."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    n = results["n_intersections"]
    intersections = [f"Int {i+1}\n(λ={results['arrivals'][i]:.2f})" for i in range(n)]

    # Compare green splits
    ax = axes[0]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w/2, results["uniform_green"], w, label="Uniform", color="lightcoral")
    ax.bar(x + w/2, results["optimal_green"], w, label="Optimised", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(intersections, fontsize=8)
    ax.set_ylabel("Green Time (s)")
    ax.set_title("Green Phase Allocation")
    ax.legend()

    # Delay comparison
    ax = axes[1]
    ax.bar(["Uniform", "Optimised"],
           [results["uniform_delay"], results["optimal_delay"]],
           color=["lightcoral", "seagreen"])
    ax.set_ylabel("Total Corridor Delay (veh·s)")
    ax.set_title(f"Total Delay ({results['improvement_pct']:.1f}% reduction)")

    # Sensitivity sweep for busiest intersection
    ax = axes[2]
    busiest = np.argmax(results["arrivals"])
    greens, delays = results["sweep_results"][busiest]
    ax.plot(greens, delays, "b-", linewidth=2)
    ax.axvline(results["optimal_green"][busiest], color="g", linestyle="--",
               label=f"Optimal ({results['optimal_green'][busiest]:.0f}s)")
    ax.axvline(results["uniform_green"][busiest], color="r", linestyle="--",
               label=f"Uniform ({results['uniform_green'][busiest]:.0f}s)")
    ax.set_xlabel("Green Time (s)")
    ax.set_ylabel("Total Corridor Delay")
    ax.set_title(f"Sensitivity: Intersection {busiest+1} (busiest)")
    ax.legend()

    plt.suptitle("Signal Timing Optimisation: Union St Corridor", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
