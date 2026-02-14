"""Animated multi-agent traffic simulation on Aberdeen city centre streets."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import osmnx as ox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"; TEXT_COLOR = "#c0caf5"; ACCENT = "#f7768e"


def speed_to_rgba(v, vm=13.4):
    t = np.clip(v / vm, 0, 1)
    if t < 0.5:
        return (1.0, t*2, 0.15, 0.85)
    return (1.0-(t-0.5)*2, 1.0, 0.15+(t-0.5)*0.3, 0.85)


def main():
    from trafficjams import aberdeen_multiagent

    print("Fetching Aberdeen street network & running simulation...")
    result = aberdeen_multiagent.simulate(n_vehicles=200, T=300, dt=1.0, n_frames=150)

    G = result["graph"]
    frames = result["frames"]
    nf = result["n_frames"]
    node_pos = result["node_pos"]

    # Get bounds
    xs = [p[0] for p in node_pos.values()]
    ys = [p[1] for p in node_pos.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 50
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    fig, ax = plt.subplots(figsize=(9, 9), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw street network
    for u, v, d in G.edges(data=True):
        x0, y0 = node_pos[u]
        x1, y1 = node_pos[v]
        ax.plot([x0, x1], [y0, y1], color="#2a2d3a", lw=1.5, zorder=1)

    # Draw intersections
    nx_arr = np.array([node_pos[n] for n in G.nodes()])
    ax.scatter(nx_arr[:, 0], nx_arr[:, 1], s=3, color="#3d4055", zorder=2)

    title = fig.text(0.5, 0.96, "Aberdeen City Centre: Multi-Agent Traffic",
                     ha="center", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    subtitle = fig.text(0.5, 0.93, "200 vehicles on real street network (OSMnx)",
                        ha="center", fontsize=9, color="#565a7a")
    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10,
                         color=TEXT_COLOR, fontfamily="monospace")

    # Vehicle scatter
    scat = ax.scatter([], [], s=12, c=[], zorder=5, edgecolors="none")

    # Stats text
    stats_text = ax.text(xmin + 20, ymax - 20, "", fontsize=8,
                         color=TEXT_COLOR, fontfamily="monospace", va="top", zorder=10)

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
        mean_spd_kmh = mean_spd * 3.6
        t_sec = frame * (300 / nf)

        time_text.set_text(f"t = {t_sec:5.0f} s  |  {n_active} vehicles  |  v\u0304 = {mean_spd_kmh:.0f} km/h")
        stats_text.set_text(f"Active: {n_active}\nMean: {mean_spd_kmh:.1f} km/h")
        return [scat, time_text, stats_text]

    anim = FuncAnimation(fig, update, frames=nf, interval=80, blit=False)
    plt.tight_layout(rect=[0, 0.04, 1, 0.92])
    sp = os.path.join(RESULTS_DIR, "anim_aberdeen_agents.gif")
    print(f"Saving {sp} ({nf} frames)...")
    anim.save(sp, writer="pillow", fps=15, dpi=100, savefig_kwargs={"facecolor": DARK_BG})
    plt.close(); print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True); main()
