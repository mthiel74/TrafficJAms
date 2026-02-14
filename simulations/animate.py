"""Animation support for IDM and Nagel-Schreckenberg models."""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def animate_idm(save_path=None):
    """Animate IDM vehicles on a circular road."""
    from trafficjams import idm

    print("Simulating IDM...")
    results = idm.simulate(n_vehicles=30, T=60.0, dt=0.1)
    positions = results["positions"]
    velocities = results["velocities"]
    road_length = results["road_length"]
    n_vehicles = results["n_vehicles"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Circular road representation
    theta = 2 * np.pi * positions[0] / road_length
    scatter = ax1.scatter(np.cos(theta), np.sin(theta),
                          c=velocities[0], cmap="RdYlGn", vmin=0, vmax=15, s=50)
    circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
    ax1.add_patch(circle)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect("equal")
    ax1.set_title("IDM: Circular Road")
    plt.colorbar(scatter, ax=ax1, label="Speed (m/s)")

    # Speed bar chart
    bars = ax2.bar(range(n_vehicles), velocities[0], color="steelblue")
    ax2.set_ylim(0, 16)
    ax2.set_xlabel("Vehicle")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_title("Vehicle Speeds")
    time_text = ax2.text(0.02, 0.95, "", transform=ax2.transAxes)

    def update(frame):
        theta = 2 * np.pi * positions[frame] / road_length
        scatter.set_offsets(np.column_stack([np.cos(theta), np.sin(theta)]))
        scatter.set_array(velocities[frame])
        for bar, v in zip(bars, velocities[frame]):
            bar.set_height(v)
        time_text.set_text(f"t = {results['times'][frame]:.1f}s")
        return scatter, *bars, time_text

    anim = FuncAnimation(fig, update, frames=len(results["times"]),
                         interval=50, blit=False)
    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="pillow", fps=20)
        print(f"  -> Saved {save_path}")
        plt.close()
    else:
        plt.show()


def animate_nasch(save_path=None):
    """Animate Nagel-Schreckenberg space-time diagram building up."""
    from trafficjams import nagel_schreckenberg

    print("Simulating NaSch...")
    results = nagel_schreckenberg.simulate(road_length=200, n_vehicles=40, T=150)
    spacetime = results["spacetime"].astype(float)
    spacetime[spacetime < 0] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    display = np.full_like(spacetime, np.nan)

    im = ax.imshow(display, aspect="auto", cmap="RdYlGn",
                   interpolation="nearest", vmin=0, vmax=results["v_max"],
                   origin="lower")
    ax.set_xlabel("Cell Position")
    ax.set_ylabel("Time Step")
    ax.set_title("Nagel-Schreckenberg: Growing Space-Time Diagram")
    plt.colorbar(im, ax=ax, label="Vehicle Speed")

    def update(frame):
        display[frame] = spacetime[frame]
        im.set_data(display)
        return [im]

    anim = FuncAnimation(fig, update, frames=results["T"],
                         interval=30, blit=True)
    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer="pillow", fps=30)
        print(f"  -> Saved {save_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Animate traffic models")
    parser.add_argument("model", choices=["idm", "nasch"],
                        help="Model to animate")
    parser.add_argument("--save", metavar="PATH",
                        help="Save animation to file (GIF)")
    args = parser.parse_args()

    if args.model == "idm":
        animate_idm(save_path=args.save)
    elif args.model == "nasch":
        animate_nasch(save_path=args.save)


if __name__ == "__main__":
    main()
