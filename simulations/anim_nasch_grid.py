"""NaSch parameter sweep: 4 values of p_slow in a grid animation."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"; ROAD_COLOR = "#2a2d3a"; TEXT_COLOR = "#c0caf5"; ACCENT = "#f7768e"

def main():
    from trafficjams import nagel_schreckenberg
    L, nv, vm = 200, 50, 5
    p_values = [0.0, 0.15, 0.3, 0.5]
    T = 180

    print("Simulating NaSch at 4 p_slow values...")
    sims = []
    for p in p_values:
        r = nagel_schreckenberg.simulate(road_length=L, n_vehicles=nv, v_max=vm, p_slow=p, T=T)
        sims.append(r["spacetime"])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), facecolor=DARK_BG)
    fig.suptitle("Nagel-Schreckenberg: Effect of Randomisation p",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=0.97)

    displays = []
    imgs = []
    hlines = []
    for idx, (ax, p) in enumerate(zip(axes.flat, p_values)):
        ax.set_facecolor(DARK_BG)
        ax.set_xlim(0, L); ax.set_ylim(0, T)
        ax.set_title(f"p = {p}", fontsize=11, color=TEXT_COLOR, pad=3)
        ax.tick_params(colors="#565a7a", labelsize=6)
        for s in ax.spines.values(): s.set_color("#2a2d3a")
        if idx >= 2: ax.set_xlabel("Cell", color="#565a7a", fontsize=8)
        if idx % 2 == 0: ax.set_ylabel("Time", color="#565a7a", fontsize=8)

        d = np.full((T, L), np.nan)
        im = ax.imshow(d, aspect="auto", cmap="RdYlGn", interpolation="nearest",
                       vmin=0, vmax=vm, origin="lower", extent=[0,L,0,T])
        hl = ax.axhline(y=0, color=ACCENT, lw=1, ls="--")
        displays.append(d); imgs.append(im); hlines.append(hl)

    time_text = fig.text(0.5, 0.01, "", ha="center", fontsize=10,
                         color=TEXT_COLOR, fontfamily="monospace")

    def update(frame):
        for idx in range(4):
            row = sims[idx][frame].astype(float)
            row[row < 0] = np.nan
            displays[idx][frame] = row
            imgs[idx].set_data(displays[idx])
            hlines[idx].set_ydata([frame, frame])
        time_text.set_text(f"t = {frame:3d}")
        return imgs + hlines + [time_text]

    anim = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    sp = os.path.join(RESULTS_DIR, "anim_nasch_grid.gif")
    print(f"Saving {sp} ({T} frames)...")
    anim.save(sp, writer="pillow", fps=20, dpi=90, savefig_kwargs={"facecolor": DARK_BG})
    plt.close(); print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True); main()
