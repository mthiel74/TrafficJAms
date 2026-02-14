"""Payne-Whitham second-order traffic model."""

import numpy as np
import matplotlib.pyplot as plt


def Ve(rho, rho_max, v_max):
    """Equilibrium speed-density relation (Greenshields)."""
    return v_max * (1 - rho / rho_max)


def simulate(L=10.0, nx=200, T=0.3, rho_max=150.0, v_max=30.0, tau=0.01, c0=5.0):
    """Simulate stop-and-go waves on a merge zone.

    Returns dict with x, t, density, velocity arrays.
    """
    dx = L / nx
    dt = 0.3 * dx / (v_max + c0)  # CFL-like condition
    nt = int(T / dt)
    x = np.linspace(0, L, nx)

    # Initial condition: uniform with perturbation at merge zone
    rho = np.ones(nx) * 40.0
    rho[(x > 4) & (x < 6)] = 90.0  # merge zone congestion
    v = Ve(rho, rho_max, v_max).copy()

    density_history = [rho.copy()]
    velocity_history = [v.copy()]
    times = [0.0]

    for n in range(nt):
        rho_new = rho.copy()
        v_new = v.copy()

        for i in range(1, nx - 1):
            # Continuity equation
            flux_r = rho[i] * v[i]
            flux_l = rho[i - 1] * v[i - 1]
            rho_new[i] = rho[i] - dt / dx * (flux_r - flux_l)

            # Momentum equation
            if rho[i] > 1e-6:
                convection = -v[i] * (v[i] - v[i - 1]) / dx
                relaxation = (Ve(rho[i], rho_max, v_max) - v[i]) / tau
                pressure = -c0**2 / rho[i] * (rho[i + 1] - rho[i]) / dx
                v_new[i] = v[i] + dt * (convection + relaxation + pressure)

        rho_new = np.clip(rho_new, 0.1, rho_max)
        v_new = np.clip(v_new, 0, v_max)
        rho = rho_new
        v = v_new

        if n % max(1, nt // 50) == 0:
            density_history.append(rho.copy())
            velocity_history.append(v.copy())
            times.append((n + 1) * dt)

    return {"x": x, "t": np.array(times),
            "density": np.array(density_history),
            "velocity": np.array(velocity_history),
            "rho_max": rho_max, "v_max": v_max}


def plot(results, save_path="results/payne_whitham.png"):
    """Plot density and velocity evolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.pcolormesh(results["x"], results["t"],
                         results["density"], shading="auto", cmap="hot_r")
    ax1.set_xlabel("Position (km)")
    ax1.set_ylabel("Time (h)")
    ax1.set_title("Payne-Whitham: Density (merge zone)")
    plt.colorbar(im1, ax=ax1, label="Density (veh/km)")

    im2 = ax2.pcolormesh(results["x"], results["t"],
                         results["velocity"], shading="auto", cmap="RdYlGn")
    ax2.set_xlabel("Position (km)")
    ax2.set_ylabel("Time (h)")
    ax2.set_title("Payne-Whitham: Velocity")
    plt.colorbar(im2, ax=ax2, label="Speed (km/h)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
