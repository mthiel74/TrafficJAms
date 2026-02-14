"""Dynamic Traffic Assignment with time-varying demand."""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx
from trafficjams.network_assignment import bpr_cost, bpr_integral


def simulate(n_periods=12, period_length=5):
    """Time-varying traffic assignment on Aberdeen network.

    Demand varies over time periods (e.g., 5-min intervals during rush hour).
    Each period solved as a static assignment with carry-over congestion.
    """
    # Simplified network (same as static)
    nodes = {
        0: "Bridge of Don", 1: "King St", 2: "Mounthooly",
        3: "Union/Market", 4: "Bridge of Dee",
        5: "Anderson Dr N", 6: "Anderson Dr S", 7: "A90 North",
    }

    edges = [
        (7, 0, 5, 2000), (0, 1, 4, 1500), (1, 2, 3, 1200),
        (2, 3, 2, 1000), (3, 4, 5, 1200), (7, 5, 6, 1800),
        (5, 6, 4, 1600), (6, 4, 3, 1400), (5, 2, 3, 800),
        (0, 5, 5, 900),
    ]

    n_edges = len(edges)
    fft = np.array([e[2] for e in edges], dtype=float)
    cap = np.array([e[3] for e in edges], dtype=float)

    # Paths (same as static model)
    paths = [[0, 1, 2, 3, 4], [5, 6, 7], [0, 9, 6, 7], [5, 8, 3, 4]]
    n_paths = len(paths)

    Delta = np.zeros((n_edges, n_paths))
    for p, path_edges in enumerate(paths):
        for e in path_edges:
            Delta[e, p] = 1

    # Time-varying demand: morning rush hour profile
    t = np.arange(n_periods)
    base_demand = 3000
    demand_profile = base_demand * (0.3 + 0.7 * np.exp(-0.5 * ((t - n_periods * 0.4) / (n_periods * 0.2)) ** 2))

    all_link_flows = np.zeros((n_periods, n_edges))
    all_path_flows = np.zeros((n_periods, n_paths))
    all_link_costs = np.zeros((n_periods, n_edges))
    residual_flow = np.zeros(n_edges)  # carry-over congestion

    for period in range(n_periods):
        demand = demand_profile[period]
        effective_cap = np.maximum(cap - residual_flow * 0.3, cap * 0.5)

        def objective(pf):
            lf = Delta @ pf + residual_flow * 0.2
            return sum(bpr_integral(lf[e], fft[e], effective_cap[e]) for e in range(n_edges))

        constraints = [{"type": "eq", "fun": lambda f: np.sum(f) - demand}]
        bounds = [(0, demand) for _ in range(n_paths)]
        x0 = np.ones(n_paths) * demand / n_paths

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        pf = result.x
        lf = Delta @ pf
        lc = np.array([bpr_cost(lf[e], fft[e], cap[e]) for e in range(n_edges)])

        all_path_flows[period] = pf
        all_link_flows[period] = lf
        all_link_costs[period] = lc
        residual_flow = lf * 0.5  # half of flow carries over as residual

    return {
        "n_periods": n_periods, "period_length": period_length,
        "demand_profile": demand_profile,
        "link_flows": all_link_flows, "link_costs": all_link_costs,
        "path_flows": all_path_flows,
        "edges": edges, "nodes": nodes, "fft": fft, "cap": cap,
    }


def plot(results, save_path="results/dynamic_assignment.png"):
    """Plot time-varying flows and costs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    periods = np.arange(results["n_periods"]) * results["period_length"]

    # Demand profile
    ax = axes[0, 0]
    ax.plot(periods, results["demand_profile"], "b-o", linewidth=2)
    ax.fill_between(periods, results["demand_profile"], alpha=0.2)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Total Demand (veh/period)")
    ax.set_title("Morning Rush Hour Demand Profile")
    ax.grid(True, alpha=0.3)

    # Path flows over time
    ax = axes[0, 1]
    for i in range(results["path_flows"].shape[1]):
        ax.plot(periods, results["path_flows"][:, i], "-o", label=f"Path {i+1}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Path Flow")
    ax.set_title("Path Flow Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Key link flows over time
    ax = axes[1, 0]
    key_links = [0, 1, 5, 6]  # A90, King St, Anderson Dr
    edge_names = ["A90→Bridge of Don", "Bridge of Don→King St",
                  "A90→Anderson Dr N", "Anderson Dr N→S"]
    for i, (idx, name) in enumerate(zip(key_links, edge_names)):
        ax.plot(periods, results["link_flows"][:, idx], "-o", label=name)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Link Flow")
    ax.set_title("Key Link Flows Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Travel cost heatmap
    ax = axes[1, 1]
    im = ax.imshow(results["link_costs"].T, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xlabel("Time Period")
    ax.set_ylabel("Link Index")
    ax.set_title("Link Travel Costs Over Time")
    plt.colorbar(im, ax=ax, label="Travel Cost")

    plt.suptitle("Dynamic Traffic Assignment: Morning Rush Hour", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
