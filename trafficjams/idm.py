"""Intelligent Driver Model (IDM) — microscopic car-following."""

import numpy as np
import matplotlib.pyplot as plt


def simulate(n_vehicles=50, road_length=1000.0, T=120.0, dt=0.1,
             v0=15.0, s0=2.0, T_headway=1.5, a=1.0, b=1.5, delta=4):
    """Simulate vehicles on a circular road. Shows spontaneous jam formation.

    Returns dict with time, positions, velocities arrays.
    """
    nt = int(T / dt)

    # Evenly spaced initial positions with small perturbation
    spacing = road_length / n_vehicles
    positions = np.array([i * spacing for i in range(n_vehicles)], dtype=float)
    velocities = np.ones(n_vehicles) * v0
    # Perturbation on one vehicle
    velocities[0] *= 0.5

    pos_history = [positions.copy()]
    vel_history = [velocities.copy()]
    times = [0.0]

    for step in range(nt):
        # Compute gaps (circular road)
        gaps = np.zeros(n_vehicles)
        for i in range(n_vehicles):
            leader = (i + 1) % n_vehicles
            gaps[i] = (positions[leader] - positions[i]) % road_length - 5.0  # vehicle length
        gaps = np.maximum(gaps, 0.1)

        # Velocity differences
        dv = np.zeros(n_vehicles)
        for i in range(n_vehicles):
            leader = (i + 1) % n_vehicles
            dv[i] = velocities[i] - velocities[leader]

        # IDM acceleration
        s_star = s0 + velocities * T_headway + velocities * dv / (2 * np.sqrt(a * b))
        s_star = np.maximum(s_star, s0)
        accel = a * (1 - (velocities / v0) ** delta - (s_star / gaps) ** 2)

        # Update
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
            "n_vehicles": n_vehicles}


def plot(results, save_path="results/idm_trajectories.png"):
    """Plot vehicle trajectories and speed profiles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Trajectories (space-time)
    for i in range(min(results["n_vehicles"], 50)):
        ax1.scatter(results["times"], results["positions"][:, i],
                    c=results["velocities"][:, i], cmap="RdYlGn",
                    s=0.5, vmin=0, vmax=15)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("IDM: Vehicle Trajectories (circular road)")

    # Speed over time for a few vehicles
    for i in [0, 10, 25, 40]:
        if i < results["n_vehicles"]:
            ax2.plot(results["times"], results["velocities"][:, i],
                     label=f"Vehicle {i}", alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("IDM: Speed Profiles")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
