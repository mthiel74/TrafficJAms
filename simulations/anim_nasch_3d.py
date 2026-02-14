"""NaSch highway from a low-angle 3D perspective view."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"; TEXT_COLOR = "#c0caf5"; ACCENT = "#f7768e"

def speed_to_color(v, vm):
    t = np.clip(v / vm, 0, 1)
    if t < 0.5: return (1.0, t*2, 0.15, 0.9)
    return (1.0-(t-0.5)*2, 1.0, 0.15+(t-0.5)*0.3, 0.9)

def main():
    from trafficjams import nagel_schreckenberg
    L, nv, vm, T = 80, 25, 5, 200

    print("Simulating NaSch for 3D view...")
    r = nagel_schreckenberg.simulate(road_length=L, n_vehicles=nv, v_max=vm, p_slow=0.3, T=T)
    st = r["spacetime"]

    fig = plt.figure(figsize=(10, 5), facecolor=DARK_BG)
    ax = fig.add_subplot(111, projection='3d', facecolor=DARK_BG)
    ax.set_xlim(0, L); ax.set_ylim(-3, 3); ax.set_zlim(0, 2)
    ax.view_init(elev=25, azim=-75)
    ax.set_axis_off()
    ax.set_facecolor(DARK_BG)

    # Road surface
    road_x = np.array([0, L, L, 0])
    road_y = np.array([-1.5, -1.5, 1.5, 1.5])
    road_z = np.array([0, 0, 0, 0])
    road = Poly3DCollection([list(zip(road_x, road_y, road_z))],
                             facecolors='#2a2d3a', edgecolors='#3d4055', linewidths=0.5)
    ax.add_collection3d(road)

    # Lane markings
    for x in range(0, L, 4):
        ax.plot([x, x+2], [0, 0], [0.001, 0.001], color='#4a4d60', lw=0.8)

    # Car bars (3D boxes represented as vertical bars)
    car_bars = []
    for i in range(nv):
        b, = ax.plot([], [], [], 's', markersize=0, color='#9ece6a', zorder=5)
        car_bars.append(b)

    title = fig.text(0.5, 0.93, "Nagel-Schreckenberg: 3D Highway View",
                     ha="center", fontsize=13, fontweight="bold", color=TEXT_COLOR)
    time_text = fig.text(0.5, 0.05, "", ha="center", fontsize=10,
                         color=TEXT_COLOR, fontfamily="monospace")

    def update(frame):
        row = st[frame]
        idx = 0
        for cell in range(L):
            if row[cell] >= 0 and idx < nv:
                car_bars[idx].set_data_3d([cell], [0], [0.5])
                car_bars[idx].set_markersize(5)
                car_bars[idx].set_color(speed_to_color(row[cell], vm))
                idx += 1
        for i in range(idx, nv):
            car_bars[i].set_markersize(0)

        # Slowly rotate camera
        ax.view_init(elev=20 + 8*np.sin(frame/40), azim=-75 + frame*0.15)
        time_text.set_text(f"t = {frame:3d}")
        return car_bars + [time_text]

    anim = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    sp = os.path.join(RESULTS_DIR, "anim_nasch_3d.gif")
    print(f"Saving {sp} ({T} frames)...")
    anim.save(sp, writer="pillow", fps=18, dpi=90, savefig_kwargs={"facecolor": DARK_BG})
    plt.close(); print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True); main()
