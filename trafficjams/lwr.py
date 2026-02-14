"""LWR (Lighthill-Whitham-Richards) model with Godunov scheme."""

import numpy as np
import matplotlib.pyplot as plt


def greenshields(rho, rho_max, v_max):
    """Greenshields fundamental diagram: v = v_max * (1 - rho/rho_max)."""
    return rho * v_max * (1 - rho / rho_max)


def godunov_flux(rho_l, rho_r, rho_max, v_max):
    """Godunov numerical flux for the LWR equation."""
    rho_crit = rho_max / 2
    q = lambda r: greenshields(r, rho_max, v_max)
    if rho_l <= rho_r:
        if rho_l >= rho_crit:
            return q(rho_l)
        elif rho_r <= rho_crit:
            return q(rho_r)
        else:
            return q(rho_crit)
    else:
        if q(rho_l) <= q(rho_r):
            return q(rho_l)
        else:
            return q(rho_l) if rho_l >= rho_crit else max(q(rho_l), q(rho_r))


def simulate(L=10.0, nx=200, T=0.5, rho_max=150.0, v_max=30.0):
    """Simulate jam propagation on a corridor using the Godunov scheme.

    Returns dict with x, t, density arrays.
    """
    dx = L / nx
    dt = 0.5 * dx / v_max  # CFL condition
    nt = int(T / dt)
    x = np.linspace(0, L, nx)

    # Initial condition: high density block in the middle (jam)
    rho = np.ones(nx) * 20.0
    rho[(x > 3) & (x < 5)] = 120.0

    history = [rho.copy()]
    times = [0.0]

    for n in range(nt):
        rho_new = rho.copy()
        for i in range(1, nx - 1):
            fl = godunov_flux(rho[i - 1], rho[i], rho_max, v_max)
            fr = godunov_flux(rho[i], rho[i + 1], rho_max, v_max)
            rho_new[i] = rho[i] - dt / dx * (fr - fl)
        rho_new = np.clip(rho_new, 0, rho_max)
        rho = rho_new
        if n % max(1, nt // 50) == 0:
            history.append(rho.copy())
            times.append((n + 1) * dt)

    return {"x": x, "t": np.array(times), "density": np.array(history),
            "rho_max": rho_max, "v_max": v_max, "dx": dx, "dt": dt}


def plot(results, save_path="results/lwr_density.png"):
    """Plot space-time density evolution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Space-time diagram
    im = ax1.pcolormesh(results["x"], results["t"],
                        results["density"], shading="auto", cmap="hot_r")
    ax1.set_xlabel("Position (km)")
    ax1.set_ylabel("Time (h)")
    ax1.set_title("LWR Model: Density Evolution (A90-like corridor)")
    plt.colorbar(im, ax=ax1, label="Density (veh/km)")

    # Density snapshots
    n_snaps = min(5, len(results["t"]))
    indices = np.linspace(0, len(results["t"]) - 1, n_snaps, dtype=int)
    for i in indices:
        ax2.plot(results["x"], results["density"][i],
                 label=f't={results["t"][i]:.3f}h')
    ax2.set_xlabel("Position (km)")
    ax2.set_ylabel("Density (veh/km)")
    ax2.set_title("Density Profiles at Different Times")
    ax2.legend()
    ax2.set_ylim(0, results["rho_max"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
