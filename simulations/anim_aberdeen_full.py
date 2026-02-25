"""Animated multi-agent traffic simulation covering all of Aberdeen.

Produces a compact GIF (small figure, low DPI) so the output stays
manageable for display.
"""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"
TEXT_COLOR = "#c0caf5"


def speed_to_rgba(v, vm=13.4):
    """Map speed to red-yellow-green colour."""
    t = np.clip(v / vm, 0, 1)
    if t < 0.5:
        return (1.0, t * 2, 0.15, 0.9)
    return (1.0 - (t - 0.5) * 2, 1.0, 0.15 + (t - 0.5) * 0.3, 0.9)


def main():
    from trafficjams import aberdeen_full_multiagent

    n_vehicles = 800
    n_frames = 200

    print(f"Fetching full Aberdeen street network & simulating {n_vehicles} vehicles …")
    result = aberdeen_full_multiagent.simulate(
        n_vehicles=n_vehicles,
        T=400,
        dt=1.0,
        n_frames=n_frames,
    )

    G = result["graph"]
    frames = result["frames"]
    nf = result["n_frames"]
    node_pos = result["node_pos"]

    # Compute bounds
    xs = [p[0] for p in node_pos.values()]
    ys = [p[1] for p in node_pos.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 200
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    # --- Small figure for compact output ---
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw streets
    for u, v, d in G.edges(data=True):
        x0, y0 = node_pos[u]
        x1, y1 = node_pos[v]
        ax.plot([x0, x1], [y0, y1], color="#2a2d3a", lw=0.4, zorder=1)

    # Title / subtitle
    fig.text(0.5, 0.96, "Aberdeen: Full City Traffic Simulation",
             ha="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.93, f"{n_vehicles} vehicles on real OSM network",
             ha="center", fontsize=7, color="#565a7a")
    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=7,
                         color=TEXT_COLOR, fontfamily="monospace")

    # Vehicle scatter
    scat = ax.scatter([], [], s=4, c=[], zorder=5, edgecolors="none")

    # Stats overlay
    stats_text = ax.text(
        xmin + 60, ymax - 60, "", fontsize=6,
        color=TEXT_COLOR, fontfamily="monospace", va="top", zorder=10,
    )

    def update(frame):
        f = frames[frame]
        pos = f["positions"]
        spd = f["speeds"]

        if pos:
            xy = np.array(pos)
            colors = [speed_to_rgba(s) for s in spd]
            scat.set_offsets(xy)
            scat.set_color(colors)
        else:
            scat.set_offsets(np.empty((0, 2)))

        n_active = len(pos)
        mean_spd = np.mean(spd) if spd else 0
        mean_kmh = mean_spd * 3.6
        t_sec = frame * (400 / nf)

        time_text.set_text(
            f"t = {t_sec:5.0f} s  |  {n_active} vehicles  |  v\u0304 = {mean_kmh:.0f} km/h"
        )
        stats_text.set_text(f"Active: {n_active}\nMean: {mean_kmh:.1f} km/h")
        return [scat, time_text, stats_text]

    anim = FuncAnimation(fig, update, frames=nf, interval=80, blit=False)
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])

    sp = os.path.join(RESULTS_DIR, "anim_aberdeen_full.gif")
    print(f"Saving {sp} ({nf} frames) …")
    anim.save(sp, writer="pillow", fps=15, dpi=72,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"Done -> {os.path.getsize(sp) / 1e6:.1f} MB")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
