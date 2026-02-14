"""Animated dynamic traffic assignment: flow pulses through Aberdeen network."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"; TEXT_COLOR = "#c0caf5"; ACCENT = "#f7768e"

def main():
    from trafficjams import dynamic_assignment
    print("Running dynamic traffic assignment...")
    r = dynamic_assignment.simulate(n_periods=24, period_length=5)

    nodes = r["nodes"]
    edges = r["edges"]
    link_flows = r["link_flows"]  # (n_periods, n_edges)
    demand = r["demand_profile"]
    n_periods = r["n_periods"]

    # Build graph for layout
    G = nx.DiGraph()
    pos = {0: (2,3), 1: (3,2.5), 2: (3,1.5), 3: (3,0.5),
           4: (2,-0.5), 5: (1,2), 6: (1,0.5), 7: (0,2)}
    for idx, name in nodes.items():
        G.add_node(idx, label=name)
    for u, v, fft, cap in edges:
        G.add_edge(u, v)

    labels = {n: nodes[n] for n in G.nodes()}
    max_flow = link_flows.max()

    fig, (ax_net, ax_dem) = plt.subplots(1, 2, figsize=(12, 5.5),
                                          gridspec_kw={"width_ratios": [1.5, 1]},
                                          facecolor=DARK_BG)
    for ax in [ax_net, ax_dem]:
        ax.set_facecolor(DARK_BG)

    ax_net.axis("off")
    fig.suptitle("Dynamic Traffic Assignment: Morning Rush Hour",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, y=0.97)

    # Static node drawing
    nx.draw_networkx_nodes(G, pos, ax=ax_net, node_size=400, node_color="#1a1b26",
                           edgecolors="#7aa2f7", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()},
                            ax=ax_net, font_size=8, font_color=TEXT_COLOR)
    # Labels below
    for n, (x, y) in pos.items():
        ax_net.text(x, y-0.25, nodes[n], ha="center", fontsize=6, color="#565a7a")

    # Demand bar
    ax_dem.set_xlim(0, n_periods)
    ax_dem.set_ylim(0, demand.max() * 1.2)
    ax_dem.set_xlabel("Time Period", color="#565a7a", fontsize=9)
    ax_dem.set_ylabel("Demand", color="#565a7a", fontsize=9)
    ax_dem.tick_params(colors="#565a7a", labelsize=7)
    for s in ax_dem.spines.values(): s.set_color("#2a2d3a")
    ax_dem.set_title("Demand Profile", fontsize=10, color=TEXT_COLOR)

    demand_bars = ax_dem.bar(range(n_periods), demand, color="#2a2d3a",
                              edgecolor="#3d4055", width=0.8)
    cursor = ax_dem.axvline(x=0, color=ACCENT, lw=2, ls="--")

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=10,
                         color=TEXT_COLOR, fontfamily="monospace")

    # Edge artists (will be redrawn each frame)
    edge_artists = []

    def update(frame):
        # Remove old edges
        for ea in edge_artists:
            ea.remove()
        edge_artists.clear()

        period = frame % n_periods
        flows = link_flows[period]

        # Draw edges with flow-based width and colour
        edge_list = list(G.edges())
        widths = [1 + 5 * flows[i] / max(max_flow, 1) for i in range(len(edge_list))]
        colors = [flows[i] / max(max_flow, 1) for i in range(len(edge_list))]

        ec = nx.draw_networkx_edges(G, pos, ax=ax_net, width=widths,
                                     edge_color=colors, edge_cmap=plt.cm.YlOrRd,
                                     edge_vmin=0, edge_vmax=1,
                                     connectionstyle="arc3,rad=0.1")
        if ec is not None:
            if hasattr(ec, '__iter__'):
                edge_artists.extend(ec)
            else:
                edge_artists.append(ec)

        # Update demand highlight
        for i, bar in enumerate(demand_bars):
            bar.set_facecolor(ACCENT if i == period else "#2a2d3a")
        cursor.set_xdata([period, period])

        t_min = period * r["period_length"]
        time_text.set_text(f"Period {period+1}/{n_periods}  |  t = {t_min} min")
        return edge_artists + list(demand_bars) + [cursor, time_text]

    # Loop 2x through all periods
    total_frames = n_periods * 2
    anim = FuncAnimation(fig, update, frames=total_frames, interval=300, blit=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    sp = os.path.join(RESULTS_DIR, "anim_network_flow.gif")
    print(f"Saving {sp} ({total_frames} frames)...")
    anim.save(sp, writer="pillow", fps=4, dpi=100, savefig_kwargs={"facecolor": DARK_BG})
    plt.close(); print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True); main()
