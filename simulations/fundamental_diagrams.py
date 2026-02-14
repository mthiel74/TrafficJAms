"""Plot flow-density fundamental diagrams from simulation data."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def fd_lwr():
    """Theoretical Greenshields fundamental diagram."""
    from trafficjams.lwr import greenshields
    rho = np.linspace(0, 150, 200)
    flow = greenshields(rho, 150, 30)
    speed = 30 * (1 - rho / 150)
    return rho, flow, speed, "LWR (Greenshields)"


def fd_idm():
    """Extract fundamental diagram from IDM simulation sweeps."""
    from trafficjams import idm
    densities, flows, speeds = [], [], []

    for n_veh in range(5, 100, 5):
        r = idm.simulate(n_vehicles=n_veh, road_length=1000, T=80, dt=0.1)
        # Use last quarter of simulation (steady state)
        n_steady = len(r["times"]) * 3 // 4
        v_avg = r["velocities"][n_steady:].mean()
        rho = n_veh / 1.0  # vehicles per km (1 km road)
        densities.append(rho)
        speeds.append(v_avg * 3.6)  # m/s to km/h
        flows.append(rho * v_avg * 3.6)

    return np.array(densities), np.array(flows), np.array(speeds), "IDM"


def fd_bando():
    """Extract fundamental diagram from Bando OVM sweeps."""
    from trafficjams import bando
    densities, flows, speeds = [], [], []

    for n_veh in range(5, 80, 5):
        r = bando.simulate(n_vehicles=n_veh, road_length=800, T=150, dt=0.1)
        n_steady = len(r["times"]) * 3 // 4
        v_avg = r["velocities"][n_steady:].mean()
        rho = n_veh / 0.8
        densities.append(rho)
        speeds.append(v_avg * 3.6)
        flows.append(rho * v_avg * 3.6)

    return np.array(densities), np.array(flows), np.array(speeds), "Bando OVM"


def fd_nasch():
    """Extract fundamental diagram from NaSch sweeps."""
    from trafficjams import nagel_schreckenberg
    densities, flows, speeds = [], [], []

    for n_veh in range(10, 250, 15):
        r = nagel_schreckenberg.simulate(road_length=500, n_vehicles=n_veh, T=200)
        st = r["spacetime"]
        # Average flow: count vehicle movements in steady state
        occupied = st[100:] >= 0
        speed_vals = st[100:].copy().astype(float)
        speed_vals[~occupied] = 0
        avg_speed = speed_vals.sum() / max(occupied.sum(), 1)
        rho = n_veh / 500
        densities.append(rho)
        speeds.append(avg_speed)
        flows.append(rho * avg_speed)

    return np.array(densities), np.array(flows), np.array(speeds), "NaSch"


def plot_all(save_path=None):
    save_path = save_path or os.path.join(RESULTS_DIR, "fundamental_diagrams.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    datasets = [fd_lwr(), fd_idm(), fd_bando(), fd_nasch()]

    for ax, (rho, flow, speed, name) in zip(axes.flat, datasets):
        color = "steelblue"
        ax.scatter(rho, flow, c=color, s=15, alpha=0.7)
        if name == "LWR (Greenshields)":
            ax.plot(rho, flow, color=color, linewidth=2)
        ax.set_xlabel("Density")
        ax.set_ylabel("Flow")
        ax.set_title(f"{name}: Fundamental Diagram")
        ax.grid(True, alpha=0.3)

        # Inset: speed-density
        ax2 = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        ax2.scatter(rho, speed, c="coral", s=10, alpha=0.7)
        if name == "LWR (Greenshields)":
            ax2.plot(rho, speed, color="coral", linewidth=1.5)
        ax2.set_xlabel("Density", fontsize=7)
        ax2.set_ylabel("Speed", fontsize=7)
        ax2.tick_params(labelsize=6)

    plt.suptitle("Fundamental Diagrams from Simulation Data", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")
    return save_path


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_all()
