"""Multi-lane (2-lane) Nagel-Schreckenberg cellular automaton."""

import numpy as np
import matplotlib.pyplot as plt


def simulate(road_length=500, n_vehicles=150, v_max=5, p_slow=0.3, T=300):
    """2-lane NaSch model with lane-changing rules.

    Lane change criteria (symmetric):
    - Incentive: gap in current lane < velocity (would need to brake)
    - Safety: gap in target lane (both ahead and behind) is sufficient
    """
    rng = np.random.default_rng(42)
    n_per_lane = n_vehicles // 2

    # Initialize two lanes
    lanes = []
    for _ in range(2):
        cells = rng.choice(road_length, size=n_per_lane, replace=False)
        cells.sort()
        vels = rng.integers(0, v_max + 1, size=n_per_lane)
        lanes.append({"pos": cells.copy(), "vel": vels.copy()})

    spacetime = np.zeros((T, 2, road_length), dtype=int)
    spacetime[:] = -1
    lane_changes = np.zeros(T, dtype=int)

    for t in range(T):
        # Record state
        for lane_idx in range(2):
            for i, pos in enumerate(lanes[lane_idx]["pos"]):
                spacetime[t, lane_idx, pos] = lanes[lane_idx]["vel"][i]

        # Lane changing phase
        changes = 0
        for lane_idx in range(2):
            other = 1 - lane_idx
            pos = lanes[lane_idx]["pos"]
            vel = lanes[lane_idx]["vel"]
            other_pos = set(lanes[other]["pos"])
            n = len(pos)

            to_switch = []
            for i in range(n):
                # Gap ahead in current lane
                sorted_idx = np.argsort(pos)
                my_sorted = np.searchsorted(pos[sorted_idx], pos[i])
                leader_idx = sorted_idx[(my_sorted + 1) % n]
                gap_current = (pos[leader_idx] - pos[i] - 1) % road_length

                # Would need to brake?
                if gap_current >= vel[i] + 1:
                    continue  # no incentive

                # Check safety in other lane
                p = pos[i]
                # Gap ahead in other lane
                gap_ahead = 0
                for d in range(1, v_max + 2):
                    if (p + d) % road_length not in other_pos:
                        gap_ahead += 1
                    else:
                        break

                # Gap behind in other lane
                gap_behind = 0
                for d in range(1, v_max + 2):
                    if (p - d) % road_length not in other_pos:
                        gap_behind += 1
                    else:
                        break

                if gap_ahead > vel[i] and gap_behind > v_max and p not in other_pos:
                    to_switch.append(i)

            # Execute lane changes
            for i in sorted(to_switch, reverse=True):
                p, v = pos[i], vel[i]
                # Move to other lane
                lanes[other]["pos"] = np.append(lanes[other]["pos"], p)
                lanes[other]["vel"] = np.append(lanes[other]["vel"], v)
                lanes[lane_idx]["pos"] = np.delete(pos, i)
                lanes[lane_idx]["vel"] = np.delete(vel, i)
                pos = lanes[lane_idx]["pos"]
                vel = lanes[lane_idx]["vel"]
                other_pos.add(p)
                changes += 1

        lane_changes[t] = changes

        # NaSch update for each lane
        for lane_idx in range(2):
            pos = lanes[lane_idx]["pos"]
            vel = lanes[lane_idx]["vel"]
            n = len(pos)
            if n == 0:
                continue

            # Sort
            order = np.argsort(pos)
            pos = pos[order]
            vel = vel[order]

            # Acceleration
            vel = np.minimum(vel + 1, v_max)

            # Braking
            gaps = np.zeros(n, dtype=int)
            for i in range(n):
                leader = (i + 1) % n
                gaps[i] = (pos[leader] - pos[i] - 1) % road_length
            vel = np.minimum(vel, gaps)

            # Randomisation
            slow = rng.random(n) < p_slow
            vel[slow] = np.maximum(vel[slow] - 1, 0)

            # Movement
            pos = (pos + vel) % road_length

            lanes[lane_idx]["pos"] = pos
            lanes[lane_idx]["vel"] = vel

    return {"spacetime": spacetime, "road_length": road_length,
            "n_vehicles": n_vehicles, "v_max": v_max, "T": T,
            "lane_changes": lane_changes}


def plot(results, save_path="results/nasch_2lane.png"):
    """Plot 2-lane space-time diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [2, 2, 1]})

    for lane_idx, ax in enumerate(axes[:2]):
        st = results["spacetime"][:, lane_idx].astype(float)
        st[st < 0] = np.nan
        im = ax.imshow(st, aspect="auto", cmap="RdYlGn",
                       interpolation="nearest", vmin=0, vmax=results["v_max"],
                       origin="lower")
        ax.set_xlabel("Cell Position")
        ax.set_ylabel("Time Step")
        ax.set_title(f"Lane {lane_idx + 1}")

    plt.colorbar(im, ax=axes[1], label="Vehicle Speed")

    # Lane change frequency
    axes[2].plot(results["lane_changes"], range(results["T"]), "b-", alpha=0.5)
    axes[2].set_xlabel("Lane Changes")
    axes[2].set_ylabel("Time Step")
    axes[2].set_title("Lane Changes")

    fig.suptitle(f"2-Lane Nagel-Schreckenberg (N={results['n_vehicles']}, L={results['road_length']})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
