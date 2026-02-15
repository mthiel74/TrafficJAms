"""Side-by-side MP4: IDM circular (left) + Aberdeen multi-agent (right)."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"; ROAD_COLOR = "#2a2d3a"; ROAD_EDGE = "#3d4055"
TEXT_COLOR = "#c0caf5"; ACCENT = "#f7768e"


def speed_to_color(v, vm):
    t = np.clip(v / vm, 0, 1)
    if t < 0.5:
        return (1.0, t*2, 0.15)
    return (1.0-(t-0.5)*2, 1.0, 0.15+(t-0.5)*0.3)


def speed_to_rgba(v, vm=13.4):
    t = np.clip(v / vm, 0, 1)
    if t < 0.5:
        return (1.0, t*2, 0.15, 0.85)
    return (1.0-(t-0.5)*2, 1.0, 0.15+(t-0.5)*0.3, 0.85)


def run_idm(nv, L, T, dt, v0, nf):
    vl = 5.0; spf = max(1, int(T/dt/nf)); sp = L/nv
    pos = np.array([i*sp for i in range(nv)], dtype=float)
    vel = np.ones(nv)*v0*0.95; vel[0] = 0.0; vel[1] = v0*0.3
    s0, Th, a, b, d = 2.0, 1.0, 1.0, 1.5, 4
    ph, vh, th = [pos.copy()], [vel.copy()], [0.0]; t = 0.0
    for _ in range(nf-1):
        for __ in range(spf):
            gaps = np.empty(nv); dv_ = np.empty(nv)
            for i in range(nv):
                l = (i+1) % nv; gaps[i] = (pos[l]-pos[i]) % L - vl; dv_[i] = vel[i]-vel[l]
            gaps = np.maximum(gaps, 0.1)
            ss = np.maximum(s0+vel*Th+vel*dv_/(2*np.sqrt(a*b)), s0)
            acc = a*(1-(vel/v0)**d-(ss/gaps)**2)
            vel = np.maximum(vel+acc*dt, 0.0); pos = (pos+vel*dt) % L; t += dt
        ph.append(pos.copy()); vh.append(vel.copy()); th.append(t)
    return np.array(ph), np.array(vh), np.array(th)


def main():
    from trafficjams import aberdeen_multiagent

    nf = 200  # common frame count
    nv = 25; L = 250.0; vm = 12.0

    print("Simulating IDM circular road...")
    idm_pos, idm_vel, idm_times = run_idm(nv, L, 80.0, 0.02, vm, nf)

    print("Fetching Aberdeen network & running multi-agent sim...")
    ab_result = aberdeen_multiagent.simulate(n_vehicles=200, T=300, dt=1.0, n_frames=nf)
    ab_frames = ab_result["frames"]
    ab_G = ab_result["graph"]
    ab_node_pos = ab_result["node_pos"]

    # Aberdeen bounds
    xs = [p[0] for p in ab_node_pos.values()]
    ys = [p[1] for p in ab_node_pos.values()]
    xmin, xmax = min(xs)-50, max(xs)+50
    ymin, ymax = min(ys)-50, max(ys)+50

    # --- Build figure ---
    fig = plt.figure(figsize=(14, 7), facecolor=DARK_BG)
    ax_idm = fig.add_axes([0.02, 0.05, 0.46, 0.85])
    ax_ab = fig.add_axes([0.52, 0.05, 0.46, 0.85])

    for ax in [ax_idm, ax_ab]:
        ax.set_facecolor(DARK_BG)
        ax.axis("off")

    # --- IDM ring road (left) ---
    Ro, Ri = 1.0, 0.78; Rm = (Ro+Ri)/2
    ax_idm.set_xlim(-1.5, 1.5); ax_idm.set_ylim(-1.5, 1.5)
    ax_idm.set_aspect("equal")

    theta = np.linspace(0, 2*np.pi, 200)
    ax_idm.fill_between(np.cos(theta)*Ro, np.sin(theta)*Ro,
                         np.cos(theta)*Ri*0.99, color=ROAD_COLOR, zorder=1)
    ax_idm.add_patch(plt.Circle((0, 0), Ro, fill=False, color=ROAD_EDGE, lw=2.5, zorder=2))
    ax_idm.add_patch(plt.Circle((0, 0), Ri, fill=False, color=ROAD_EDGE, lw=2.5, zorder=2))
    for i in range(60):
        a1 = i*2*np.pi/60; t_ = np.linspace(a1, a1+np.pi/60*0.6, 10)
        ax_idm.plot(np.cos(t_)*Rm, np.sin(t_)*Rm, color="#4a4d60", lw=1, zorder=2)

    cars = []
    for i in range(nv):
        b = FancyBboxPatch((0, 0), 0.055, 0.025, boxstyle="round,pad=0.005",
                            facecolor="#9ece6a", edgecolor="none", zorder=5)
        ax_idm.add_patch(b); cars.append(b)

    ax_idm.text(0, 1.35, "Intelligent Driver Model", ha="center", fontsize=13,
                fontweight="bold", color=TEXT_COLOR, zorder=10)
    ax_idm.text(0, 1.2, f"{nv} vehicles  ·  ring road", ha="center", fontsize=9,
                color="#565a7a", zorder=10)
    idm_time = ax_idm.text(0, -1.35, "", ha="center", fontsize=10,
                            color=TEXT_COLOR, fontfamily="monospace", zorder=10)

    # Speed bar legend
    for j in range(50):
        t = j/49
        c = speed_to_color(t*vm, vm)
        ax_idm.plot([-0.7+j*1.4/50], [-1.45], "s", color=c, markersize=3, zorder=10)
    ax_idm.text(-0.75, -1.45, "0", ha="right", fontsize=7, color="#565a7a")
    ax_idm.text(0.75, -1.45, f"{vm:.0f} m/s", ha="left", fontsize=7, color="#565a7a")

    # --- Aberdeen (right) ---
    ax_ab.set_xlim(xmin, xmax); ax_ab.set_ylim(ymin, ymax)
    ax_ab.set_aspect("equal")

    for u, v, d in ab_G.edges(data=True):
        x0, y0 = ab_node_pos[u]; x1, y1 = ab_node_pos[v]
        ax_ab.plot([x0, x1], [y0, y1], color="#2a2d3a", lw=1.2, zorder=1)

    nx_arr = np.array([ab_node_pos[n] for n in ab_G.nodes()])
    ax_ab.scatter(nx_arr[:, 0], nx_arr[:, 1], s=2, color="#3d4055", zorder=2)

    ax_ab.text(0.5, 0.97, "Aberdeen City Centre", transform=ax_ab.transAxes,
               ha="center", fontsize=13, fontweight="bold", color=TEXT_COLOR, zorder=10)
    ax_ab.text(0.5, 0.93, "200 vehicles  ·  real streets (OSMnx)", transform=ax_ab.transAxes,
               ha="center", fontsize=9, color="#565a7a", zorder=10)

    ab_scat = ax_ab.scatter([], [], s=10, c=[], zorder=5, edgecolors="none")
    ab_time = ax_ab.text(0.5, 0.02, "", transform=ax_ab.transAxes, ha="center",
                          fontsize=10, color=TEXT_COLOR, fontfamily="monospace", zorder=10)

    fig.text(0.5, 0.97, "TrafficJAms: Micro vs City-Scale Simulation",
             ha="center", fontsize=15, fontweight="bold", color=TEXT_COLOR)

    def update(frame):
        # IDM update
        p = idm_pos[frame]; v = idm_vel[frame]; t = idm_times[frame]
        for i in range(nv):
            angle = 2*np.pi*p[i]/L
            cx, cy = np.cos(angle)*Rm, np.sin(angle)*Rm
            rd = np.degrees(angle)-90
            tr = (mtransforms.Affine2D().translate(-0.0275, -0.0125)
                  .rotate_deg(rd).translate(cx, cy)+ax_idm.transData)
            cars[i].set_transform(tr)
            cars[i].set_facecolor(speed_to_color(v[i], vm))
        idm_time.set_text(f"t = {t:5.1f} s  |  v\u0304 = {v.mean():.1f} m/s")

        # Aberdeen update
        f = ab_frames[frame]
        pos = f["positions"]; spd = f["speeds"]
        if pos:
            xy = np.array(pos)
            colors = [speed_to_rgba(s) for s in spd]
            ab_scat.set_offsets(xy)
            ab_scat.set_color(colors)
        n_active = len(pos)
        mean_spd = np.mean(spd)*3.6 if spd else 0
        ab_time.set_text(f"{n_active} active  |  v\u0304 = {mean_spd:.0f} km/h")

        return cars + [idm_time, ab_scat, ab_time]

    anim = FuncAnimation(fig, update, frames=nf, interval=40, blit=False)
    sp = os.path.join(RESULTS_DIR, "combined_idm_aberdeen.mp4")
    print(f"Saving {sp} ({nf} frames)...")
    writer = FFMpegWriter(fps=25, metadata={"title": "TrafficJAms Combined"}, bitrate=2000)
    anim.save(sp, writer=writer, dpi=120, savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True); main()
