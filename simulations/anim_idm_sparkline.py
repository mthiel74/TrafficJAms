"""IDM circular road with rolling mean-speed sparkline overlay."""

import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"
ROAD_COLOR = "#2a2d3a"
ROAD_EDGE = "#3d4055"
TEXT_COLOR = "#c0caf5"
ACCENT = "#f7768e"

def speed_to_color(v, v_max):
    t = np.clip(v / v_max, 0, 1)
    if t < 0.5:
        return (1.0, t * 2, 0.15)
    return (1.0 - (t - 0.5) * 2, 1.0, 0.15 + (t - 0.5) * 0.3)


def run_idm(n_veh, road_length, T, dt, v0, n_frames):
    veh_len = 5.0
    steps_per_frame = max(1, int(T / dt / n_frames))
    spacing = road_length / n_veh
    pos = np.array([i * spacing for i in range(n_veh)], dtype=float)
    vel = np.ones(n_veh) * v0 * 0.95
    vel[0] = 0.0; vel[1] = v0 * 0.3
    s0, T_hw, a, b, delta = 2.0, 1.0, 1.0, 1.5, 4

    pos_h, vel_h, t_h = [pos.copy()], [vel.copy()], [0.0]
    t = 0.0
    for _ in range(n_frames - 1):
        for __ in range(steps_per_frame):
            gaps = np.empty(n_veh); dv = np.empty(n_veh)
            for i in range(n_veh):
                ldr = (i + 1) % n_veh
                gaps[i] = (pos[ldr] - pos[i]) % road_length - veh_len
                dv[i] = vel[i] - vel[ldr]
            gaps = np.maximum(gaps, 0.1)
            s_star = np.maximum(s0 + vel * T_hw + vel * dv / (2 * np.sqrt(a * b)), s0)
            acc = a * (1 - (vel / v0) ** delta - (s_star / gaps) ** 2)
            vel = np.maximum(vel + acc * dt, 0.0)
            pos = (pos + vel * dt) % road_length
            t += dt
        pos_h.append(pos.copy()); vel_h.append(vel.copy()); t_h.append(t)
    return np.array(pos_h), np.array(vel_h), np.array(t_h)


def main():
    n_frames = 200
    n_veh, road_length, v_max = 25, 250.0, 12.0

    print("Simulating IDM with sparkline...")
    positions, velocities, times = run_idm(n_veh, road_length, 80.0, 0.02, v_max, n_frames)

    R_outer, R_inner = 1.0, 0.78
    R_mid = (R_outer + R_inner) / 2

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal"); ax.axis("off")

    # Road
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(np.cos(th) * R_outer, np.sin(th) * R_outer,
                    np.cos(th) * R_inner * 0.99, color=ROAD_COLOR, zorder=1)
    ax.add_patch(plt.Circle((0, 0), R_outer, fill=False, color=ROAD_EDGE, lw=2.5, zorder=2))
    ax.add_patch(plt.Circle((0, 0), R_inner, fill=False, color=ROAD_EDGE, lw=2.5, zorder=2))
    for i in range(60):
        a1 = i * 2 * np.pi / 60
        t_ = np.linspace(a1, a1 + np.pi / 60 * 0.6, 10)
        ax.plot(np.cos(t_) * R_mid, np.sin(t_) * R_mid, color="#4a4d60", lw=1, zorder=2)

    car_bodies = []
    for i in range(n_veh):
        body = FancyBboxPatch((0, 0), 0.055, 0.025, boxstyle="round,pad=0.005",
                               facecolor="#9ece6a", edgecolor="none", zorder=5)
        ax.add_patch(body)
        car_bodies.append(body)

    ax.text(0, 1.38, "IDM: Mean Speed Sparkline", ha="center",
            fontsize=15, fontweight="bold", color=TEXT_COLOR, zorder=10)
    time_text = ax.text(0, -1.50, "", ha="center", fontsize=10,
                        color=TEXT_COLOR, fontfamily="monospace", zorder=10)

    # Sparkline inset (inside the ring)
    spark_ax = fig.add_axes([0.30, 0.32, 0.40, 0.20])  # [left, bottom, w, h]
    spark_ax.set_facecolor(DARK_BG)
    spark_ax.set_xlim(0, n_frames)
    spark_ax.set_ylim(0, v_max * 1.1)
    spark_ax.set_ylabel("v\u0304 (m/s)", fontsize=8, color="#565a7a")
    spark_ax.tick_params(colors="#565a7a", labelsize=6)
    for spine in spark_ax.spines.values():
        spine.set_color("#2a2d3a")
    spark_ax.axhline(y=v_max, color="#565a7a", ls=":", lw=0.5)
    spark_ax.set_xticks([])

    mean_speeds = []
    spark_line, = spark_ax.plot([], [], color=ACCENT, lw=1.5)
    spark_fill = None
    speed_text = spark_ax.text(0.98, 0.92, "", transform=spark_ax.transAxes,
                                ha="right", va="top", fontsize=9, color=ACCENT,
                                fontweight="bold")

    def update(frame):
        nonlocal spark_fill
        pos = positions[frame]
        vel = velocities[frame]
        t = times[frame]

        for i in range(n_veh):
            angle = 2 * np.pi * pos[i] / road_length
            cx, cy = np.cos(angle) * R_mid, np.sin(angle) * R_mid
            rot_deg = np.degrees(angle) - 90
            tr = (mtransforms.Affine2D()
                  .translate(-0.0275, -0.0125).rotate_deg(rot_deg)
                  .translate(cx, cy) + ax.transData)
            car_bodies[i].set_transform(tr)
            car_bodies[i].set_facecolor(speed_to_color(vel[i], v_max))

        # Update sparkline
        mean_speeds.append(vel.mean())
        xs = list(range(len(mean_speeds)))
        spark_line.set_data(xs, mean_speeds)
        if spark_fill is not None:
            spark_fill.remove()
        spark_fill = spark_ax.fill_between(xs, 0, mean_speeds, color=ACCENT, alpha=0.15)

        speed_text.set_text(f"{vel.mean():.1f} m/s")
        time_text.set_text(f"t = {t:5.1f} s")
        return car_bodies + [spark_line, speed_text, time_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
    save_path = os.path.join(RESULTS_DIR, "anim_idm_sparkline.gif")
    print(f"Saving {save_path} ({n_frames} frames)...")
    anim.save(save_path, writer="pillow", fps=25, dpi=100,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"Done -> {os.path.getsize(save_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
