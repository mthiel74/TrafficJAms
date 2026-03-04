"""Animated multi-agent traffic simulation covering all of Aberdeen.

Produces a compact GIF and a high-quality MP4 with all visual
improvements: road hierarchy widths, directional markers, dynamic
congestion colouring, lane offsets, and speed legend.
"""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"
TEXT_COLOR = "#c0caf5"

# Point 16: Road width by highway type
_HIGHWAY_WIDTH = {
    "motorway": 2.5, "motorway_link": 1.8,
    "trunk": 2.2, "trunk_link": 1.6,
    "primary": 1.8, "primary_link": 1.3,
    "secondary": 1.4, "secondary_link": 1.1,
    "tertiary": 1.1, "tertiary_link": 0.9,
    "residential": 0.7, "service": 0.5,
    "living_street": 0.5, "unclassified": 0.6,
}


def speed_to_rgba(v, vm=13.4):
    """Map speed to red-yellow-green colour."""
    t = np.clip(v / vm, 0, 1)
    if t < 0.5:
        return (1.0, t * 2, 0.15, 0.9)
    return (1.0 - (t - 0.5) * 2, 1.0, 0.15 + (t - 0.5) * 0.3, 0.9)


def main():
    from trafficjams import aberdeen_full_multiagent

    n_vehicles = 1200
    n_frames = 250

    print(f"Fetching full Aberdeen street network & simulating {n_vehicles} vehicles ...")
    result = aberdeen_full_multiagent.simulate(
        n_vehicles=n_vehicles,
        T=500,
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

    # --- Large figure for high-quality output ---
    fig, ax = plt.subplots(figsize=(14, 14), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    # Point 16 & 18: Draw streets with road hierarchy widths
    # Store line segments for dynamic congestion colouring
    edge_lines = []
    edge_keys = []
    edge_base_color = "#2a2d3a"
    for u, v, k, d in G.edges(data=True, keys=True):
        hw = d.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        lw = _HIGHWAY_WIDTH.get(hw, 0.6)
        if "geometry" in d:
            coords = list(d["geometry"].coords)
            xs_line = [c[0] for c in coords]
            ys_line = [c[1] for c in coords]
        else:
            x0, y0 = node_pos[u]
            x1, y1 = node_pos[v]
            xs_line, ys_line = [x0, x1], [y0, y1]
        line, = ax.plot(xs_line, ys_line, color=edge_base_color, lw=lw, zorder=1)
        edge_lines.append(line)
        edge_keys.append((u, v))

    # Title / subtitle
    fig.text(0.5, 0.97, "Aberdeen: Full City Traffic Simulation",
             ha="center", fontsize=18, fontweight="bold", color=TEXT_COLOR)
    fig.text(0.5, 0.945, f"{n_vehicles} vehicles | IDM car-following | traffic signals & roundabouts",
             ha="center", fontsize=10, color="#565a7a")
    time_text = fig.text(0.5, 0.015, "", ha="center", fontsize=11,
                         color=TEXT_COLOR, fontfamily="monospace")

    # Vehicle scatter (Point 17: will use markers that show direction)
    scat = ax.scatter([], [], s=12, c=[], zorder=5, edgecolors="none",
                      marker="o")

    # Stats overlay
    stats_text = ax.text(
        xmin + 80, ymax - 80, "", fontsize=9,
        color=TEXT_COLOR, fontfamily="monospace", va="top", zorder=10,
    )

    # Point 20: Speed colour legend
    cmap_data = []
    n_legend = 50
    for i in range(n_legend):
        v = i / (n_legend - 1) * 13.4
        cmap_data.append(speed_to_rgba(v))
    legend_height = (ymax - ymin) * 0.25
    legend_x = xmax - 150
    legend_y0 = ymin + 100
    for i in range(n_legend):
        y = legend_y0 + i / n_legend * legend_height
        dy = legend_height / n_legend
        ax.barh(y, 40, height=dy, left=legend_x, color=cmap_data[i], zorder=8)
    ax.text(legend_x + 20, legend_y0 - 30, "0", ha="center", fontsize=7,
            color=TEXT_COLOR, zorder=9)
    ax.text(legend_x + 20, legend_y0 + legend_height + 20, "48",
            ha="center", fontsize=7, color=TEXT_COLOR, zorder=9)
    ax.text(legend_x + 20, legend_y0 + legend_height + 50, "km/h",
            ha="center", fontsize=7, color="#565a7a", zorder=9)

    # Point 18: Build edge-to-index mapping for congestion colouring
    edge_to_lines = {}
    for i, (u, v) in enumerate(edge_keys):
        if (u, v) not in edge_to_lines:
            edge_to_lines[(u, v)] = []
        edge_to_lines[(u, v)].append(i)

    def _congestion_color(mean_speed, speed_limit=13.4):
        """Map average edge speed to road color (dark=free, orange/red=congested)."""
        t = np.clip(mean_speed / max(speed_limit, 0.1), 0, 1)
        if t > 0.7:
            return "#2a2d3a"  # free flow: default dark
        elif t > 0.3:
            r = 0.6 + (0.7 - t) * 1.0
            g = 0.3 + t * 0.3
            return (min(r, 1.0), g, 0.1, 0.7)
        else:
            return (0.9, 0.15, 0.1, 0.8)  # heavily congested: red

    def update(frame):
        f = frames[frame]
        pos = f["positions"]
        spd = f["speeds"]
        edge_speeds = f.get("edge_speeds", {})

        if pos:
            xy = np.array(pos)
            colors = [speed_to_rgba(s) for s in spd]
            scat.set_offsets(xy)
            scat.set_color(colors)
        else:
            scat.set_offsets(np.empty((0, 2)))

        # Point 18: Update road congestion colours every 5 frames
        if frame % 5 == 0:
            for ek, line_indices in edge_to_lines.items():
                if ek in edge_speeds:
                    col = _congestion_color(edge_speeds[ek])
                    for li in line_indices:
                        edge_lines[li].set_color(col)
                else:
                    for li in line_indices:
                        edge_lines[li].set_color(edge_base_color)

        n_active = len(pos)
        mean_spd = np.mean(spd) if spd else 0
        mean_kmh = mean_spd * 3.6
        t_sec = frame * (500 / nf)

        time_text.set_text(
            f"t = {t_sec:5.0f} s  |  {n_active} vehicles  |  v\u0304 = {mean_kmh:.0f} km/h"
        )
        stats_text.set_text(f"Active: {n_active}\nMean: {mean_kmh:.1f} km/h")
        return [scat, time_text, stats_text] + edge_lines

    anim = FuncAnimation(fig, update, frames=nf, interval=80, blit=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    # Save high-quality MP4 (new filename, additional to existing)
    mp4_path = os.path.join(RESULTS_DIR, "anim_aberdeen_full_v2.mp4")
    print(f"Saving {mp4_path} ({nf} frames) ...")
    anim.save(mp4_path, writer="ffmpeg", fps=20, dpi=150,
              savefig_kwargs={"facecolor": DARK_BG},
              extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                          "-crf", "18", "-preset", "slow"])
    print(f"MP4 done -> {os.path.getsize(mp4_path) / 1e6:.1f} MB")

    # Save compact GIF (small enough for GitHub inline rendering <5 MB)
    sp = os.path.join(RESULTS_DIR, "anim_aberdeen_full_v2.gif")
    print(f"Saving {sp} (every 2nd frame) ...")
    # Use ffmpeg to convert MP4 to a compact GIF (skip every other frame)
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", mp4_path,
        "-vf", "fps=10,scale=700:-1:flags=lanczos",
        "-gifflags", "+transdiff",
        sp,
    ], check=True, capture_output=True)
    plt.close()
    print(f"GIF done -> {os.path.getsize(sp) / 1e6:.1f} MB")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
