"""Optimal Velocity Model (Bando OVM) — shows Hopf bifurcation / phantom jams."""

import numpy as np
import matplotlib.pyplot as plt


def optimal_velocity(s, v_max=15.0, s_c=25.0):
    """Optimal velocity function V_opt(s) = v_max * tanh(s/s_c - 2) + tanh(2)) / (1 + tanh(2))."""
    return v_max * (np.tanh(s / s_c - 2) + np.tanh(2)) / (1 + np.tanh(2))


def simulate(n_vehicles=40, road_length=800.0, T=200.0, dt=0.1,
             kappa=1.0, v_max=15.0, s_c=25.0):
    """Simulate Bando OVM on a circular road.

    Returns dict with time, positions, velocities arrays.
    """
    nt = int(T / dt)
    spacing = road_length / n_vehicles
    positions = np.array([i * spacing for i in range(n_vehicles)], dtype=float)
    velocities = np.ones(n_vehicles) * optimal_velocity(spacing, v_max, s_c)

    # Perturbation
    positions[0] -= 5.0
    positions = positions % road_length

    pos_history = [positions.copy()]
    vel_history = [velocities.copy()]
    times = [0.0]

    for step in range(nt):
        # Compute gaps
        gaps = np.zeros(n_vehicles)
        for i in range(n_vehicles):
            leader = (i + 1) % n_vehicles
            gaps[i] = (positions[leader] - positions[i]) % road_length

        # Bando acceleration
        v_opt = optimal_velocity(gaps, v_max, s_c)
        accel = kappa * (v_opt - velocities)

        velocities = np.maximum(velocities + accel * dt, 0)
        positions = (positions + velocities * dt) % road_length

        if step % max(1, nt // 200) == 0:
            pos_history.append(positions.copy())
            vel_history.append(velocities.copy())
            times.append((step + 1) * dt)

    return {"times": np.array(times),
            "positions": np.array(pos_history),
            "velocities": np.array(vel_history),
            "road_length": road_length,
            "n_vehicles": n_vehicles,
            "kappa": kappa}


def plot(results, save_path="results/bando_ovm.png"):
    """Plot OVM results showing phantom jams."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectories
    for i in range(results["n_vehicles"]):
        ax1.scatter(results["times"], results["positions"][:, i],
                    c=results["velocities"][:, i], cmap="RdYlGn",
                    s=0.5, vmin=0, vmax=15)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.set_title(f"Bando OVM: Phantom Jams (κ={results['kappa']})")

    # Speed distribution over time
    mean_v = results["velocities"].mean(axis=1)
    std_v = results["velocities"].std(axis=1)
    ax2.plot(results["times"], mean_v, label="Mean speed")
    ax2.fill_between(results["times"], mean_v - std_v, mean_v + std_v,
                     alpha=0.3, label="±1 std")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Speed Statistics Over Time")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
