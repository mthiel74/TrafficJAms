"""Nagel-Schreckenberg cellular automaton model."""

import numpy as np
import matplotlib.pyplot as plt


def simulate(road_length=500, n_vehicles=100, v_max=5, p_slow=0.3, T=300):
    """Simulate NaSch model on a 1D periodic lattice.

    Returns dict with space-time occupancy and velocity arrays.
    """
    rng = np.random.default_rng(42)

    # Place vehicles randomly
    all_cells = rng.choice(road_length, size=n_vehicles, replace=False)
    all_cells.sort()

    positions = all_cells.copy()
    velocities = rng.integers(0, v_max + 1, size=n_vehicles)

    # Space-time diagram storage
    spacetime = np.zeros((T, road_length), dtype=int)  # -1 = empty, >=0 = velocity
    spacetime[:] = -1

    for t in range(T):
        # Record
        for i, pos in enumerate(positions):
            spacetime[t, pos] = velocities[i]

        # 1. Acceleration
        velocities = np.minimum(velocities + 1, v_max)

        # 2. Braking: find gap to next vehicle
        sorted_idx = np.argsort(positions)
        positions = positions[sorted_idx]
        velocities = velocities[sorted_idx]

        gaps = np.zeros(n_vehicles, dtype=int)
        for i in range(n_vehicles):
            leader = (i + 1) % n_vehicles
            gaps[i] = (positions[leader] - positions[i] - 1) % road_length

        velocities = np.minimum(velocities, gaps)

        # 3. Randomisation
        slow_mask = rng.random(n_vehicles) < p_slow
        velocities[slow_mask] = np.maximum(velocities[slow_mask] - 1, 0)

        # 4. Movement
        positions = (positions + velocities) % road_length

    return {"spacetime": spacetime, "road_length": road_length,
            "n_vehicles": n_vehicles, "v_max": v_max, "T": T}


def plot(results, save_path="results/nasch_spacetime.png"):
    """Plot space-time diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create display array: empty cells white, occupied cells colored by speed
    st = results["spacetime"].astype(float)
    st[st < 0] = np.nan

    im = ax.imshow(st, aspect="auto", cmap="RdYlGn",
                   interpolation="nearest", vmin=0, vmax=results["v_max"],
                   origin="lower")
    ax.set_xlabel("Cell Position")
    ax.set_ylabel("Time Step")
    ax.set_title(f"Nagel-Schreckenberg: Space-Time Diagram "
                 f"(N={results['n_vehicles']}, L={results['road_length']})")
    plt.colorbar(im, ax=ax, label="Vehicle Speed")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
