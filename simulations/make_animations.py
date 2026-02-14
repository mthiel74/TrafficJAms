"""High-quality animations for TrafficJAms models.

Generates polished GIFs suitable for README embedding.
Usage: python -m simulations.make_animations [idm|nasch|lwr|bando|all]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Circle, Wedge, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ── Shared styling ─────────────────────────────────────────────────────────────
DARK_BG = "#0f1117"
CARD_BG = "#1a1b26"
ROAD_COLOR = "#2a2d3a"
ROAD_EDGE = "#3d4055"
TEXT_COLOR = "#c0caf5"
ACCENT = "#f7768e"
GREEN = "#9ece6a"
YELLOW = "#e0af68"
BLUE = "#7aa2f7"
CYAN = "#7dcfff"

def speed_to_color(v, v_max):
    """Map speed to a color: red (stopped) -> yellow -> green (free flow)."""
    t = np.clip(v / v_max, 0, 1)
    # Red -> Yellow -> Green
    if t < 0.5:
        r, g, b = 1.0, t * 2, 0.15
    else:
        r, g, b = 1.0 - (t - 0.5) * 2, 1.0, 0.15 + (t - 0.5) * 0.3
    return (r, g, b)

def speed_to_color_array(v, v_max):
    """Vectorised speed-to-color."""
    t = np.clip(v / v_max, 0, 1)
    colors = np.zeros((len(t), 3))
    low = t < 0.5
    colors[low, 0] = 1.0
    colors[low, 1] = t[low] * 2
    colors[low, 2] = 0.15
    colors[~low, 0] = 1.0 - (t[~low] - 0.5) * 2
    colors[~low, 1] = 1.0
    colors[~low, 2] = 0.15 + (t[~low] - 0.5) * 0.3
    return colors


# ── 1. IDM Circular Road ──────────────────────────────────────────────────────

def _run_idm_jam(n_veh, road_length, T, dt, v0, s0, T_hw, a, b, delta,
                  n_frames):
    """Run IDM simulation with a hard-brake perturbation, returning one
    snapshot per animation frame so every frame shows dynamics."""
    veh_len = 5.0
    steps_per_frame = max(1, int(T / dt / n_frames))
    spacing = road_length / n_veh
    positions = np.array([i * spacing for i in range(n_veh)], dtype=float)
    velocities = np.ones(n_veh) * v0 * 0.95  # start near free-flow

    # Hard brake: vehicle 0 stops, vehicle 1 slows sharply
    velocities[0] = 0.0
    velocities[1] = v0 * 0.3

    pos_hist = [positions.copy()]
    vel_hist = [velocities.copy()]
    t_hist = [0.0]

    t = 0.0
    frame = 0
    while frame < n_frames - 1:
        for _ in range(steps_per_frame):
            gaps = np.empty(n_veh)
            dv = np.empty(n_veh)
            for i in range(n_veh):
                leader = (i + 1) % n_veh
                gaps[i] = (positions[leader] - positions[i]) % road_length - veh_len
                dv[i] = velocities[i] - velocities[leader]
            gaps = np.maximum(gaps, 0.1)
            s_star = s0 + velocities * T_hw + velocities * dv / (2 * np.sqrt(a * b))
            s_star = np.maximum(s_star, s0)
            accel = a * (1 - (velocities / v0) ** delta - (s_star / gaps) ** 2)
            velocities = np.maximum(velocities + accel * dt, 0.0)
            positions = (positions + velocities * dt) % road_length
            t += dt
        frame += 1
        pos_hist.append(positions.copy())
        vel_hist.append(velocities.copy())
        t_hist.append(t)

    return np.array(pos_hist), np.array(vel_hist), np.array(t_hist)


def animate_idm_circular(save_path=None, n_frames=250):
    """Cars driving on a circular road with IDM dynamics.

    Shows phantom jam forming and propagating backward.
    """
    print("  Simulating IDM (circular road)...")
    n_veh = 25
    road_length = 250.0
    v_max = 12.0
    positions, velocities, times = _run_idm_jam(
        n_veh=n_veh, road_length=road_length, T=80.0, dt=0.02,
        v0=v_max, s0=2.0, T_hw=1.0, a=1.0, b=1.5, delta=4,
        n_frames=n_frames)

    # Road geometry
    R_outer = 1.0
    R_inner = 0.78
    R_mid = (R_outer + R_inner) / 2
    lane_w = R_outer - R_inner

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw the road (circular ring)
    theta_ring = np.linspace(0, 2 * np.pi, 200)
    ax.fill_between(np.cos(theta_ring) * R_outer, np.sin(theta_ring) * R_outer,
                    np.cos(theta_ring) * R_inner * 0.99,  # avoid z-fighting
                    color=ROAD_COLOR, zorder=1)
    outer_circle = plt.Circle((0, 0), R_outer, fill=False, color=ROAD_EDGE, linewidth=2.5, zorder=2)
    inner_circle = plt.Circle((0, 0), R_inner, fill=False, color=ROAD_EDGE, linewidth=2.5, zorder=2)
    ax.add_patch(outer_circle)
    ax.add_patch(inner_circle)

    # Dashed centre line
    for i in range(60):
        a1 = i * 2 * np.pi / 60
        a2 = a1 + np.pi / 60 * 0.6
        th = np.linspace(a1, a2, 10)
        ax.plot(np.cos(th) * R_mid, np.sin(th) * R_mid, color="#4a4d60",
                linewidth=1, zorder=2)

    # Car patches (rectangles oriented along the road)
    car_length = 0.055
    car_width = 0.025
    cars = []
    car_bodies = []
    for i in range(n_veh):
        body = FancyBboxPatch((0, 0), car_length, car_width,
                              boxstyle="round,pad=0.005",
                              facecolor=GREEN, edgecolor="none", zorder=5)
        ax.add_patch(body)
        car_bodies.append(body)

    # Title and info text
    title_text = ax.text(0, 1.38, "Intelligent Driver Model",
                         ha="center", va="center", fontsize=16, fontweight="bold",
                         color=TEXT_COLOR, zorder=10)
    subtitle_text = ax.text(0, 1.22, f"{n_veh} vehicles  ·  circular road",
                            ha="center", va="center", fontsize=10, color="#565a7a", zorder=10)
    time_text = ax.text(0, -1.32, "", ha="center", va="center", fontsize=11,
                        color=TEXT_COLOR, zorder=10,
                        fontfamily="monospace")

    # Speed bar (legend)
    bar_y = -1.45
    for j in range(50):
        t = j / 49
        c = speed_to_color(t * v_max, v_max)
        ax.plot([-0.7 + j * 1.4 / 50], [bar_y], "s", color=c, markersize=4, zorder=10)
    ax.text(-0.75, bar_y, "0", ha="right", va="center", fontsize=8, color="#565a7a")
    ax.text(0.75, bar_y, f"{v_max:.0f} m/s", ha="left", va="center", fontsize=8, color="#565a7a")

    def update(frame):
        pos = positions[frame]
        vel = velocities[frame]
        t = times[frame]

        for i in range(n_veh):
            angle = 2 * np.pi * pos[i] / road_length
            # Car centre on the road
            cx = np.cos(angle) * R_mid
            cy = np.sin(angle) * R_mid
            # Rotation: tangent to circle (perpendicular to radius)
            rot_deg = np.degrees(angle) - 90

            import matplotlib.transforms as mtransforms
            tr = (mtransforms.Affine2D()
                  .translate(-car_length / 2, -car_width / 2)
                  .rotate_deg(rot_deg)
                  .translate(cx, cy)
                  + ax.transData)
            car_bodies[i].set_transform(tr)
            car_bodies[i].set_facecolor(speed_to_color(vel[i], v_max))

        time_text.set_text(f"t = {t:5.1f} s")
        return car_bodies + [time_text]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=33, blit=False)

    save_path = save_path or os.path.join(RESULTS_DIR, "anim_idm_circular.gif")
    print(f"  Saving {save_path}  ({n_frames} frames)...")
    anim.save(save_path, writer="pillow", fps=25, dpi=100,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"  Done  -> {os.path.getsize(save_path) / 1e6:.1f} MB")
    return save_path


# ── 2. Nagel-Schreckenberg Highway ───────────────────────────────────────────

def animate_nasch_highway(save_path=None, n_frames=250):
    """Top-down view of cars on a straight highway, NaSch cellular automaton."""
    from trafficjams import nagel_schreckenberg

    print("  Simulating Nagel-Schreckenberg...")
    road_length = 120
    n_veh = 30
    v_max = 5
    T = n_frames
    results = nagel_schreckenberg.simulate(road_length=road_length, n_vehicles=n_veh,
                                            v_max=v_max, p_slow=0.3, T=T)
    spacetime = results["spacetime"]

    # Viewport: show a sliding window of the road
    view_cells = 60  # how many cells visible at once

    fig, (ax_road, ax_st) = plt.subplots(2, 1, figsize=(10, 5),
                                          gridspec_kw={"height_ratios": [1, 1.8]},
                                          facecolor=DARK_BG)
    for ax in [ax_road, ax_st]:
        ax.set_facecolor(DARK_BG)

    # ─ Top panel: road view ─
    ax_road.set_xlim(-1, view_cells + 1)
    ax_road.set_ylim(-1.5, 1.5)
    ax_road.axis("off")

    # Draw road surface
    road_rect = FancyBboxPatch((-0.5, -0.6), view_cells + 1, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=ROAD_COLOR, edgecolor=ROAD_EDGE,
                                linewidth=1.5, zorder=1)
    ax_road.add_patch(road_rect)

    # Lane markings (dashed)
    for x in range(0, view_cells, 3):
        ax_road.plot([x, x + 1.5], [0, 0], color="#4a4d60", linewidth=1, zorder=2)

    # Car patches for road view
    car_patches_road = []
    for i in range(n_veh):
        car = plt.Rectangle((0, -0.3), 0.8, 0.6,
                              facecolor=GREEN, edgecolor="none", zorder=5, visible=False,
                              joinstyle="round")
        ax_road.add_patch(car)
        car_patches_road.append(car)

    road_title = ax_road.text(view_cells / 2, 1.2, "Nagel-Schreckenberg Cellular Automaton",
                               ha="center", fontsize=13, fontweight="bold",
                               color=TEXT_COLOR, zorder=10)
    time_txt = ax_road.text(view_cells - 1, -1.2, "", ha="right", fontsize=10,
                            color=TEXT_COLOR, fontfamily="monospace", zorder=10)

    # ─ Bottom panel: space-time diagram building up ─
    ax_st.set_xlim(0, road_length)
    ax_st.set_ylim(0, n_frames)
    ax_st.set_xlabel("Cell Position", color=TEXT_COLOR, fontsize=9)
    ax_st.set_ylabel("Time Step", color=TEXT_COLOR, fontsize=9)
    ax_st.tick_params(colors="#565a7a", labelsize=8)
    for spine in ax_st.spines.values():
        spine.set_color("#2a2d3a")

    # Pre-render space-time as image
    st_display = np.full((n_frames, road_length), np.nan)
    st_img = ax_st.imshow(st_display, aspect="auto", cmap="RdYlGn",
                           interpolation="nearest", vmin=0, vmax=v_max,
                           origin="lower", extent=[0, road_length, 0, n_frames],
                           zorder=1)
    time_line = ax_st.axhline(y=0, color=ACCENT, linewidth=1.5, linestyle="--", zorder=5)

    def update(frame):
        t = frame
        row = spacetime[t]

        # Update road view
        # Find camera centre: follow the average position of vehicles
        occupied = np.where(row >= 0)[0]
        cam_centre = int(np.mean(occupied)) if len(occupied) > 0 else road_length // 2
        cam_start = cam_centre - view_cells // 2

        for i, car in enumerate(car_patches_road):
            car.set_visible(False)

        car_idx = 0
        for cell in range(road_length):
            if row[cell] >= 0:
                display_x = (cell - cam_start) % road_length
                if 0 <= display_x < view_cells:
                    if car_idx < n_veh:
                        car_patches_road[car_idx].set_xy((display_x - 0.4, -0.3))
                        car_patches_road[car_idx].set_facecolor(
                            speed_to_color(row[cell], v_max))
                        car_patches_road[car_idx].set_visible(True)
                        car_idx += 1

        time_txt.set_text(f"t = {t:3d}")

        # Update space-time diagram
        st_row = row.astype(float)
        st_row[st_row < 0] = np.nan
        st_display[t] = st_row
        st_img.set_data(st_display)
        time_line.set_ydata([t, t])

        return car_patches_road + [time_txt, st_img, time_line]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=60, blit=False)
    plt.tight_layout(pad=1.0)

    save_path = save_path or os.path.join(RESULTS_DIR, "anim_nasch_highway.gif")
    print(f"  Saving {save_path}  ({n_frames} frames)...")
    anim.save(save_path, writer="pillow", fps=18, dpi=120,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"  Done  -> {os.path.getsize(save_path) / 1e6:.1f} MB")
    return save_path


# ── 3. LWR Shockwave ─────────────────────────────────────────────────────────

def animate_lwr_shockwave(save_path=None, n_frames=200):
    """Animated density profile showing shockwave propagation."""
    from trafficjams import lwr

    print("  Simulating LWR...")
    results = lwr.simulate(L=10.0, nx=300, T=0.5, rho_max=150.0, v_max=30.0)
    x = results["x"]
    density = results["density"]
    rho_max = results["rho_max"]
    v_max_val = results["v_max"]

    total = len(results["t"])
    frame_idx = np.linspace(0, total - 1, n_frames, dtype=int)

    fig, (ax_road, ax_dens) = plt.subplots(2, 1, figsize=(10, 5.5),
                                            gridspec_kw={"height_ratios": [1, 2]},
                                            facecolor=DARK_BG)
    for ax in [ax_road, ax_dens]:
        ax.set_facecolor(DARK_BG)

    # ─ Top: road visualisation with density-colored segments ─
    ax_road.set_xlim(0, 10)
    ax_road.set_ylim(-1, 1.5)
    ax_road.axis("off")
    ax_road.text(5, 1.2, "LWR Shockwave Propagation", ha="center",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)

    # Road surface
    road_bg = FancyBboxPatch((0.1, -0.4), 9.8, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=ROAD_COLOR, edgecolor=ROAD_EDGE,
                              linewidth=1.5, zorder=1)
    ax_road.add_patch(road_bg)

    n_seg = 80
    seg_w = 9.8 / n_seg
    road_segments = []
    for i in range(n_seg):
        seg = plt.Rectangle((0.1 + i * seg_w, -0.35), seg_w * 0.95, 0.7,
                              facecolor=GREEN, edgecolor="none", zorder=3)
        ax_road.add_patch(seg)
        road_segments.append(seg)

    time_text_lwr = ax_road.text(9.8, -0.8, "", ha="right", fontsize=10,
                                  color=TEXT_COLOR, fontfamily="monospace", zorder=10)

    # ─ Bottom: density profile ─
    ax_dens.set_xlim(0, 10)
    ax_dens.set_ylim(0, 155)
    ax_dens.set_xlabel("Position along A90 corridor (km)", color=TEXT_COLOR, fontsize=10)
    ax_dens.set_ylabel("Density (veh/km)", color=TEXT_COLOR, fontsize=10)
    ax_dens.tick_params(colors="#565a7a", labelsize=8)
    for spine in ax_dens.spines.values():
        spine.set_color("#2a2d3a")
    ax_dens.axhline(y=rho_max / 2, color="#565a7a", linestyle=":", linewidth=0.8, zorder=1)
    ax_dens.text(0.3, rho_max / 2 + 3, "critical density", fontsize=8, color="#565a7a")

    line, = ax_dens.plot([], [], color=ACCENT, linewidth=2.5, zorder=5)
    fill = ax_dens.fill_between(x, 0, 0, color=ACCENT, alpha=0.15, zorder=3)

    def update(frame):
        nonlocal fill
        fi = frame_idx[frame]
        rho = density[fi]
        t = results["t"][fi]

        line.set_data(x, rho)
        fill.remove()
        fill = ax_dens.fill_between(x, 0, rho, color=ACCENT, alpha=0.15, zorder=3)

        # Update road segments color
        for i, seg in enumerate(road_segments):
            xi = 0.1 + (i + 0.5) * seg_w
            idx = int(xi / 10 * len(x))
            idx = min(idx, len(rho) - 1)
            local_rho = rho[idx]
            # Map density to color
            t_val = local_rho / rho_max
            if t_val < 0.3:
                c = GREEN
            elif t_val < 0.6:
                c = YELLOW
            else:
                c = ACCENT
            seg.set_facecolor(c)
            seg.set_alpha(0.3 + 0.7 * t_val)

        time_text_lwr.set_text(f"t = {t:.3f} h")
        return [line, fill, time_text_lwr] + road_segments

    anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
    plt.tight_layout(pad=1.0)

    save_path = save_path or os.path.join(RESULTS_DIR, "anim_lwr_shockwave.gif")
    print(f"  Saving {save_path}  ({n_frames} frames)...")
    anim.save(save_path, writer="pillow", fps=25, dpi=120,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"  Done  -> {os.path.getsize(save_path) / 1e6:.1f} MB")
    return save_path


# ── 4. Bando Phantom Jam ─────────────────────────────────────────────────────

def _run_bando_jam(n_veh, road_length, T, dt, kappa, v_max, s_c, n_frames):
    """Run Bando OVM with parameters tuned for Hopf bifurcation.

    Place equilibrium headway at the inflection point of V_opt (= 2*s_c)
    and choose kappa below the critical stability threshold.
    """
    from trafficjams.bando import optimal_velocity

    steps_per_frame = max(1, int(T / dt / n_frames))
    spacing = road_length / n_veh
    positions = np.array([i * spacing for i in range(n_veh)], dtype=float)
    v_eq = optimal_velocity(spacing, v_max, s_c)
    velocities = np.ones(n_veh) * v_eq

    # Perturbation: displace one vehicle backward (creates a larger gap behind,
    # smaller gap ahead → braking → cascading instability)
    positions[0] = (positions[0] - spacing * 0.4) % road_length

    pos_hist = [positions.copy()]
    vel_hist = [velocities.copy()]
    t_hist = [0.0]

    t = 0.0
    for frame in range(n_frames - 1):
        for _ in range(steps_per_frame):
            gaps = np.empty(n_veh)
            for i in range(n_veh):
                leader = (i + 1) % n_veh
                gaps[i] = (positions[leader] - positions[i]) % road_length
            v_opt = optimal_velocity(gaps, v_max, s_c)
            accel = kappa * (v_opt - velocities)
            velocities = np.maximum(velocities + accel * dt, 0.0)
            positions = (positions + velocities * dt) % road_length
            t += dt
        pos_hist.append(positions.copy())
        vel_hist.append(velocities.copy())
        t_hist.append(t)

    return np.array(pos_hist), np.array(vel_hist), np.array(t_hist)


def animate_bando_phantom(save_path=None, n_frames=150):
    """Bando OVM on circular road showing phantom jam emergence."""

    print("  Simulating Bando OVM...")
    # Place equilibrium at inflection: spacing = 2 * s_c
    # V'_opt at inflection = v_max / (s_c * (1 + tanh(2)))
    # Critical kappa = V'_opt / 2
    # s_c=12 => inflection at 24m, n_veh=22, L=528 => spacing=24
    # V'_opt = 14 / (12 * 1.964) = 0.594, critical = 0.297
    # Use kappa = 0.22 (well below critical => unstable)
    n_veh = 22
    road_length = 528.0
    v_max = 14.0
    s_c = 12.0
    kappa = 0.22

    positions, velocities, times = _run_bando_jam(
        n_veh=n_veh, road_length=road_length, T=120.0, dt=0.02,
        kappa=kappa, v_max=v_max, s_c=s_c, n_frames=n_frames)

    fig, (ax_ring, ax_speed) = plt.subplots(1, 2, figsize=(12, 5.5),
                                             gridspec_kw={"width_ratios": [1, 1.3]},
                                             facecolor=DARK_BG)
    for ax in [ax_ring, ax_speed]:
        ax.set_facecolor(DARK_BG)

    # ─ Left: circular road ─
    R_outer = 1.0
    R_inner = 0.80
    R_mid = (R_outer + R_inner) / 2

    ax_ring.set_xlim(-1.4, 1.4)
    ax_ring.set_ylim(-1.4, 1.4)
    ax_ring.set_aspect("equal")
    ax_ring.axis("off")

    theta_ring = np.linspace(0, 2 * np.pi, 200)
    ax_ring.fill_between(np.cos(theta_ring) * R_outer,
                          np.sin(theta_ring) * R_outer,
                          np.cos(theta_ring) * R_inner,
                          color=ROAD_COLOR, zorder=1)
    ax_ring.add_patch(plt.Circle((0, 0), R_outer, fill=False, color=ROAD_EDGE, linewidth=2, zorder=2))
    ax_ring.add_patch(plt.Circle((0, 0), R_inner, fill=False, color=ROAD_EDGE, linewidth=2, zorder=2))
    for i in range(40):
        a1 = i * 2 * np.pi / 40
        a2 = a1 + np.pi / 40 * 0.5
        th = np.linspace(a1, a2, 10)
        ax_ring.plot(np.cos(th) * R_mid, np.sin(th) * R_mid, color="#4a4d60", linewidth=0.8, zorder=2)

    ax_ring.text(0, 1.25, f"Bando OVM: Phantom Jam (\u03ba={kappa})", ha="center",
                  fontsize=13, fontweight="bold", color=TEXT_COLOR)

    car_bodies = []
    for i in range(n_veh):
        body = FancyBboxPatch((0, 0), 0.06, 0.028,
                               boxstyle="round,pad=0.005",
                               facecolor=GREEN, edgecolor="none", zorder=5)
        ax_ring.add_patch(body)
        car_bodies.append(body)

    time_text_b = ax_ring.text(0, -1.25, "", ha="center", fontsize=10,
                                color=TEXT_COLOR, fontfamily="monospace")

    # ─ Right: speed waterfall ─
    ax_speed.set_xlim(0, n_veh)
    ax_speed.set_ylim(0, v_max * 1.1)
    ax_speed.set_xlabel("Vehicle index", color=TEXT_COLOR, fontsize=10)
    ax_speed.set_ylabel("Speed (m/s)", color=TEXT_COLOR, fontsize=10)
    ax_speed.tick_params(colors="#565a7a", labelsize=8)
    for spine in ax_speed.spines.values():
        spine.set_color("#2a2d3a")
    ax_speed.axhline(y=v_max, color="#565a7a", linestyle=":", linewidth=0.8)

    bars = ax_speed.bar(range(n_veh), np.zeros(n_veh), color=GREEN,
                         edgecolor="none", width=0.8, zorder=3)

    def update(frame):
        pos = positions[frame]
        vel = velocities[frame]
        t = times[frame]

        import matplotlib.transforms as mtransforms
        for i in range(n_veh):
            angle = 2 * np.pi * pos[i] / road_length
            cx = np.cos(angle) * R_mid
            cy = np.sin(angle) * R_mid
            rot_deg = np.degrees(angle) - 90
            tr = (mtransforms.Affine2D()
                  .translate(-0.03, -0.014)
                  .rotate_deg(rot_deg)
                  .translate(cx, cy)
                  + ax_ring.transData)
            car_bodies[i].set_transform(tr)
            car_bodies[i].set_facecolor(speed_to_color(vel[i], v_max))

        # Sort vehicles by position for bar chart
        order = np.argsort(pos)
        sorted_vel = vel[order]
        for j, bar in enumerate(bars):
            bar.set_height(sorted_vel[j])
            bar.set_facecolor(speed_to_color(sorted_vel[j], v_max))

        time_text_b.set_text(f"t = {t:5.1f} s")
        return car_bodies + list(bars) + [time_text_b]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=33, blit=False)
    plt.tight_layout(pad=1.5)

    save_path = save_path or os.path.join(RESULTS_DIR, "anim_bando_phantom.gif")
    print(f"  Saving {save_path}  ({n_frames} frames)...")
    anim.save(save_path, writer="pillow", fps=20, dpi=90,
              savefig_kwargs={"facecolor": DARK_BG})
    plt.close()
    print(f"  Done  -> {os.path.getsize(save_path) / 1e6:.1f} MB")
    return save_path


# ── Main ──────────────────────────────────────────────────────────────────────

ANIMATIONS = {
    "idm":   ("IDM Circular Road",       animate_idm_circular),
    "nasch": ("NaSch Highway",            animate_nasch_highway),
    "lwr":   ("LWR Shockwave",           animate_lwr_shockwave),
    "bando": ("Bando Phantom Jam",       animate_bando_phantom),
}


def main():
    parser = argparse.ArgumentParser(description="Generate high-quality animations")
    parser.add_argument("models", nargs="*", default=["all"],
                        choices=list(ANIMATIONS.keys()) + ["all"],
                        help="Which animations to generate")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    targets = list(ANIMATIONS.keys()) if "all" in args.models else args.models

    for key in targets:
        name, func = ANIMATIONS[key]
        print(f"\n{'─' * 60}")
        print(f"Generating: {name}")
        print(f"{'─' * 60}")
        func()

    print(f"\nAll animations saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
