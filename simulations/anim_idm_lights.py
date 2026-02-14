"""IDM circular road animation with headlight/taillight glow effects."""

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
    vel[0] = 0.0
    vel[1] = v0 * 0.3
    s0, T_hw, a, b, delta = 2.0, 1.0, 1.0, 1.5, 4

    pos_h, vel_h, acc_h, t_h = [pos.copy()], [vel.copy()], [np.zeros(n_veh)], [0.0]
    t = 0.0
    for _ in range(n_frames - 1):
        for __ in range(steps_per_frame):
            gaps = np.empty(n_veh)
            dv = np.empty(n_veh)
            for i in range(n_veh):
                leader = (i + 1) % n_veh
                gaps[i] = (pos[leader] - pos[i]) % road_length - veh_len
                dv[i] = vel[i] - vel[leader]
            gaps = np.maximum(gaps, 0.1)
            s_star = np.maximum(s0 + vel * T_hw + vel * dv / (2 * np.sqrt(a * b)), s0)
            acc = a * (1 - (vel / v0) ** delta - (s_star / gaps) ** 2)
            vel = np.maximum(vel + acc * dt, 0.0)
            pos = (pos + vel * dt) % road_length
            t += dt
        pos_h.append(pos.copy())
        vel_h.append(vel.copy())
        acc_h.append(acc.copy())
        t_h.append(t)
    return np.array(pos_h), np.array(vel_h), np.array(acc_h), np.array(t_h)


def main():
    n_frames = 200
    n_veh = 25
    road_length = 250.0
    v_max = 12.0

    print("Simulating IDM with lights...")
    positions, velocities, accels, times = run_idm(n_veh, road_length, 80.0, 0.02, v_max, n_frames)

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
        a2 = a1 + np.pi / 60 * 0.6
        t_ = np.linspace(a1, a2, 10)
        ax.plot(np.cos(t_) * R_mid, np.sin(t_) * R_mid, color="#4a4d60", lw=1, zorder=2)

    car_len, car_w = 0.055, 0.025
    car_bodies = []
    # Taillight glows (red circles behind braking cars)
    tail_glows = []
    # Headlight glows (white circles in front of accelerating cars)
    head_glows = []
    for i in range(n_veh):
        tg = ax.plot([], [], 'o', color='#ff2222', markersize=0, alpha=0, zorder=4)[0]
        hg = ax.plot([], [], 'o', color='#ffffcc', markersize=0, alpha=0, zorder=4)[0]
        tail_glows.append(tg)
        head_glows.append(hg)
        body = FancyBboxPatch((0, 0), car_len, car_w, boxstyle="round,pad=0.005",
                               facecolor="#9ece6a", edgecolor="none", zorder=5)
        ax.add_patch(body)
        car_bodies.append(body)

    ax.text(0, 1.38, "IDM with Brake / Headlights", ha="center", va="center",
            fontsize=15, fontweight="bold", color=TEXT_COLOR, zorder=10)
    ax.text(0, 1.22, f"{n_veh} vehicles  ·  circular road", ha="center",
            va="center", fontsize=10, color="#565a7a", zorder=10)
    time_text = ax.text(0, -1.32, "", ha="center", fontsize=11,
                        color=TEXT_COLOR, fontfamily="monospace", zorder=10)

    # Legend
    ax.plot([-0.5], [-1.45], 'o', color='#ff2222', markersize=6, alpha=0.8)
    ax.text(-0.4, -1.45, "braking", fontsize=8, color="#ff6666", va="center")
    ax.plot([0.2], [-1.45], 'o', color='#ffffcc', markersize=6, alpha=0.8)
    ax.text(0.3, -1.45, "accelerating", fontsize=8, color="#ffffcc", va="center")

    def update(frame):
        pos = positions[frame]
        vel = velocities[frame]
        acc = accels[frame]
        t = times[frame]

        for i in range(n_veh):
            angle = 2 * np.pi * pos[i] / road_length
            cx = np.cos(angle) * R_mid
            cy = np.sin(angle) * R_mid
            rot_deg = np.degrees(angle) - 90

            tr = (mtransforms.Affine2D()
                  .translate(-car_len / 2, -car_w / 2)
                  .rotate_deg(rot_deg)
                  .translate(cx, cy) + ax.transData)
            car_bodies[i].set_transform(tr)
            car_bodies[i].set_facecolor(speed_to_color(vel[i], v_max))

            # Taillight: behind car (opposite to direction of travel)
            tail_offset = 0.04
            tx = cx - np.cos(angle) * tail_offset
            ty = cy - np.sin(angle) * tail_offset
            # Headlight: in front
            hx = cx + np.cos(angle) * tail_offset
            hy = cy + np.sin(angle) * tail_offset

            if acc[i] < -0.3:  # braking
                intensity = min(1.0, abs(acc[i]) / 2.0)
                tail_glows[i].set_data([tx], [ty])
                tail_glows[i].set_markersize(4 + 6 * intensity)
                tail_glows[i].set_alpha(0.5 + 0.5 * intensity)
                head_glows[i].set_alpha(0)
            elif acc[i] > 0.2:  # accelerating
                intensity = min(1.0, acc[i] / 1.5)
                head_glows[i].set_data([hx], [hy])
                head_glows[i].set_markersize(3 + 4 * intensity)
                head_glows[i].set_alpha(0.3 + 0.4 * intensity)
                tail_glows[i].set_alpha(0)
            else:
                tail_glows[i].set_alpha(0)
                head_glows[i].set_alpha(0)

        time_text.set_text(f"t = {t:5.1f} s")
        return car_bodies + tail_glows + head_glows + [time_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
    save_path = os.path.join(RESULTS_DIR, "anim_idm_lights.gif")
    print(f"Saving {save_path} ({n_frames} frames)...")
    anim.save(save_path, writer="pillow", fps=25, dpi=100,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"Done -> {os.path.getsize(save_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
