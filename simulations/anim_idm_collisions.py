"""IDM animation with collision/near-miss warning markers."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"; ROAD_COLOR = "#2a2d3a"; ROAD_EDGE = "#3d4055"
TEXT_COLOR = "#c0caf5"; ACCENT = "#f7768e"

def speed_to_color(v, vm):
    t = np.clip(v / vm, 0, 1)
    if t < 0.5: return (1.0, t * 2, 0.15)
    return (1.0 - (t - 0.5) * 2, 1.0, 0.15 + (t - 0.5) * 0.3)

def run_idm(n_veh, L, T, dt, v0, n_frames):
    vl = 5.0; spf = max(1, int(T / dt / n_frames))
    sp = L / n_veh
    pos = np.array([i * sp for i in range(n_veh)], dtype=float)
    vel = np.ones(n_veh) * v0 * 0.95; vel[0] = 0.0; vel[1] = v0 * 0.3
    s0, Th, a, b, d = 2.0, 1.0, 1.0, 1.5, 4
    ph, vh, gh, th = [pos.copy()], [vel.copy()], [], [0.0]
    t = 0.0
    for _ in range(n_frames - 1):
        for __ in range(spf):
            gaps = np.empty(n_veh); dv = np.empty(n_veh)
            for i in range(n_veh):
                ldr = (i + 1) % n_veh
                gaps[i] = (pos[ldr] - pos[i]) % L - vl
                dv[i] = vel[i] - vel[ldr]
            gaps = np.maximum(gaps, 0.1)
            ss = np.maximum(s0 + vel * Th + vel * dv / (2 * np.sqrt(a * b)), s0)
            acc = a * (1 - (vel / v0) ** d - (ss / gaps) ** 2)
            vel = np.maximum(vel + acc * dt, 0.0)
            pos = (pos + vel * dt) % L; t += dt
        # Record gaps at this frame
        fg = np.empty(n_veh)
        for i in range(n_veh):
            fg[i] = (pos[(i+1)%n_veh] - pos[i]) % L - vl
        ph.append(pos.copy()); vh.append(vel.copy()); gh.append(fg.copy()); th.append(t)
    # First frame gaps
    fg0 = np.empty(n_veh)
    for i in range(n_veh):
        fg0[i] = (ph[0][(i+1)%n_veh] - ph[0][i]) % L - vl
    gh.insert(0, fg0)
    return np.array(ph), np.array(vh), np.array(gh), np.array(th)


def main():
    nf, nv, L, vm = 200, 25, 250.0, 12.0
    CRITICAL_GAP = 3.0  # meters — near-miss threshold

    print("Simulating IDM with collision detection...")
    pos, vel, gaps, times = run_idm(nv, L, 80.0, 0.02, vm, nf)

    Ro, Ri = 1.0, 0.78; Rm = (Ro + Ri) / 2
    fig, ax = plt.subplots(figsize=(7, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal"); ax.axis("off")

    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(np.cos(th)*Ro, np.sin(th)*Ro, np.cos(th)*Ri*0.99, color=ROAD_COLOR, zorder=1)
    ax.add_patch(plt.Circle((0,0), Ro, fill=False, color=ROAD_EDGE, lw=2.5, zorder=2))
    ax.add_patch(plt.Circle((0,0), Ri, fill=False, color=ROAD_EDGE, lw=2.5, zorder=2))
    for i in range(60):
        a1 = i*2*np.pi/60; t_ = np.linspace(a1, a1+np.pi/60*0.6, 10)
        ax.plot(np.cos(t_)*Rm, np.sin(t_)*Rm, color="#4a4d60", lw=1, zorder=2)

    cars = []
    for i in range(nv):
        b = FancyBboxPatch((0,0), 0.055, 0.025, boxstyle="round,pad=0.005",
                            facecolor="#9ece6a", edgecolor="none", zorder=5)
        ax.add_patch(b); cars.append(b)

    # Warning markers (triangles)
    warnings = []
    for i in range(nv):
        w, = ax.plot([], [], marker='^', color='#ff0000', markersize=0, alpha=0, zorder=8)
        warnings.append(w)

    ax.text(0, 1.38, "IDM: Near-Miss Detection", ha="center", fontsize=15,
            fontweight="bold", color=TEXT_COLOR, zorder=10)
    time_text = ax.text(0, -1.32, "", ha="center", fontsize=11,
                        color=TEXT_COLOR, fontfamily="monospace", zorder=10)
    warn_counter = ax.text(0, -1.45, "", ha="center", fontsize=10,
                           color=ACCENT, fontfamily="monospace", zorder=10)
    total_warnings = [0]

    def update(frame):
        p, v, g = pos[frame], vel[frame], gaps[frame]
        t = times[frame]
        frame_warns = 0

        for i in range(nv):
            angle = 2*np.pi*p[i]/L
            cx, cy = np.cos(angle)*Rm, np.sin(angle)*Rm
            rd = np.degrees(angle) - 90
            tr = (mtransforms.Affine2D().translate(-0.0275,-0.0125)
                  .rotate_deg(rd).translate(cx,cy) + ax.transData)
            cars[i].set_transform(tr)
            cars[i].set_facecolor(speed_to_color(v[i], vm))

            if g[i] < CRITICAL_GAP:
                # Near-miss: show warning above car
                wx = np.cos(angle) * (Ro + 0.08)
                wy = np.sin(angle) * (Ro + 0.08)
                warnings[i].set_data([wx], [wy])
                intensity = 1.0 - g[i] / CRITICAL_GAP
                warnings[i].set_markersize(6 + 8 * intensity)
                warnings[i].set_alpha(0.6 + 0.4 * intensity)
                # Flash the car edge
                cars[i].set_edgecolor('#ff0000')
                cars[i].set_linewidth(1.5 * intensity)
                frame_warns += 1
            else:
                warnings[i].set_alpha(0)
                cars[i].set_edgecolor('none')

        total_warnings[0] += frame_warns
        time_text.set_text(f"t = {t:5.1f} s")
        warn_counter.set_text(f"\u26a0 {frame_warns} near-misses  |  total: {total_warnings[0]}")
        return cars + warnings + [time_text, warn_counter]

    anim = FuncAnimation(fig, update, frames=nf, interval=40, blit=False)
    sp = os.path.join(RESULTS_DIR, "anim_idm_collisions.gif")
    print(f"Saving {sp} ({nf} frames)...")
    anim.save(sp, writer="pillow", fps=25, dpi=100, savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True); main()
