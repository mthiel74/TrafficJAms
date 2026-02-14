"""Realistic Aberdeen road network for traffic assignment."""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx


# Aberdeen junction nodes with approximate coordinates (for layout)
NODES = {
    0:  ("Bridge of Don Roundabout", (1.2, 4.0)),
    1:  ("Haudagain Roundabout", (0.5, 3.2)),
    2:  ("Persley Bridge", (0.2, 3.5)),
    3:  ("King St / St Machar Dr", (1.4, 3.2)),
    4:  ("Mounthooly Roundabout", (1.2, 2.5)),
    5:  ("Berryden / Westburn", (0.7, 2.5)),
    6:  ("Union St / King St", (1.1, 1.8)),
    7:  ("Union St / Market St", (0.9, 1.6)),
    8:  ("Guild St / Bridge St", (0.8, 1.3)),
    9:  ("Anderson Drive / A90", (0.0, 2.0)),
    10: ("Anderson Drive / Queen's Rd", (0.0, 1.2)),
    11: ("Bridge of Dee", (0.5, 0.3)),
    12: ("A90 North (AWPR Junction)", (0.5, 4.5)),
    13: ("A96 / Auchmill Rd", (0.0, 3.8)),
    14: ("Beach Boulevard / Esplanade", (1.8, 2.8)),
}

# Edges: (from, to, free_flow_time_min, capacity_veh_hr, road_name)
EDGES = [
    (12, 0,  3, 3000, "A90 Parkway"),
    (12, 13, 4, 2500, "A90/A96 Link"),
    (13, 1,  3, 2000, "A96 Auchmill Rd"),
    (13, 2,  2, 1500, "Mugiemoss Rd"),
    (0,  3,  4, 1800, "King Street (north)"),
    (0,  1,  5, 1200, "Tillydrone Ave"),
    (0,  14, 3, 1000, "Beach Boulevard"),
    (1,  2,  2, 1200, "Persley Rd"),
    (1,  5,  3, 1400, "Berryden Rd"),
    (2,  9,  4, 1600, "Anderson Dr (north)"),
    (3,  4,  3, 1200, "King Street (mid)"),
    (3,  14, 2, 800,  "Linksfield Rd"),
    (4,  6,  3, 1000, "King Street (south)"),
    (4,  5,  2, 900,  "Loch St / George St"),
    (5,  6,  3, 1000, "Skene St / Rosemount"),
    (5,  9,  3, 1400, "Westburn Rd"),
    (6,  7,  2, 1200, "Union Street"),
    (7,  8,  2, 1100, "Market Street"),
    (8,  11, 5, 1500, "Holburn St / Gt Southern Rd"),
    (9,  10, 4, 2000, "Anderson Drive (south)"),
    (10, 11, 3, 1800, "Anderson Dr / Garthdee"),
    (10, 7,  4, 1000, "Queen's Rd / Albyn Pl"),
    (14, 4,  3, 800,  "Gallowgate"),
]

# OD demands (from, to, demand)
OD_PAIRS = [
    (12, 11, 4000),  # A90 North to Bridge of Dee (north-south through)
    (12, 7,  2000),  # A90 North to city centre
    (13, 7,  1500),  # A96 to city centre
    (13, 11, 1000),  # A96 to Bridge of Dee
]


def bpr_cost(flow, fft, cap, alpha=0.15, beta=4.0):
    return fft * (1 + alpha * (flow / cap) ** beta)


def bpr_integral(flow, fft, cap, alpha=0.15, beta=4.0):
    return fft * (flow + alpha * cap / (beta + 1) * (flow / cap) ** (beta + 1))


def find_paths(G, origin, destination, max_paths=5):
    """Find simple paths using k-shortest paths."""
    try:
        paths = list(nx.shortest_simple_paths(G, origin, destination, weight="free_flow"))
        return paths[:max_paths]
    except nx.NetworkXNoPath:
        return []


def simulate():
    """Solve traffic assignment on the realistic Aberdeen network."""
    G = nx.DiGraph()
    for idx, (name, pos) in NODES.items():
        G.add_node(idx, label=name, pos=pos)

    edge_list = []
    for i, (u, v, fft, cap, road) in enumerate(EDGES):
        G.add_edge(u, v, index=i, free_flow=fft, capacity=cap, road=road)
        edge_list.append((u, v, fft, cap, road))

    n_edges = len(edge_list)
    fft = np.array([e[2] for e in edge_list], dtype=float)
    cap = np.array([e[3] for e in edge_list], dtype=float)

    # Find paths for each OD pair
    all_paths = []  # list of (od_idx, node_path, edge_indices)
    od_path_groups = []  # start/end indices for each OD

    for od_idx, (orig, dest, demand) in enumerate(OD_PAIRS):
        node_paths = find_paths(G, orig, dest)
        start = len(all_paths)
        for np_ in node_paths:
            edge_indices = []
            for j in range(len(np_) - 1):
                eidx = G[np_[j]][np_[j + 1]]["index"]
                edge_indices.append(eidx)
            all_paths.append((od_idx, np_, edge_indices))
        od_path_groups.append((start, len(all_paths)))

    n_all_paths = len(all_paths)

    # Path-link incidence matrix
    Delta = np.zeros((n_edges, n_all_paths))
    for p, (_, _, edge_indices) in enumerate(all_paths):
        for e in edge_indices:
            Delta[e, p] = 1

    def objective(path_flows):
        link_flows = Delta @ path_flows
        return sum(bpr_integral(link_flows[e], fft[e], cap[e]) for e in range(n_edges))

    # Constraints: for each OD pair, path flows sum to demand
    constraints = []
    for od_idx, (orig, dest, demand) in enumerate(OD_PAIRS):
        start, end = od_path_groups[od_idx]
        constraints.append({
            "type": "eq",
            "fun": lambda f, s=start, e=end, d=demand: np.sum(f[s:e]) - d
        })

    bounds = [(0, None) for _ in range(n_all_paths)]
    x0 = np.zeros(n_all_paths)
    for od_idx, (orig, dest, demand) in enumerate(OD_PAIRS):
        start, end = od_path_groups[od_idx]
        n_p = end - start
        if n_p > 0:
            x0[start:end] = demand / n_p

    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    path_flows = np.maximum(result.x, 0)
    link_flows = Delta @ path_flows
    link_costs = np.array([bpr_cost(link_flows[e], fft[e], cap[e]) for e in range(n_edges)])

    # Add results to graph
    for i, (u, v, _, _, _) in enumerate(edge_list):
        G[u][v]["flow"] = link_flows[i]
        G[u][v]["cost"] = link_costs[i]
        G[u][v]["saturation"] = link_flows[i] / cap[i]

    return {
        "graph": G, "nodes": NODES, "edges": edge_list,
        "link_flows": link_flows, "link_costs": link_costs,
        "path_flows": path_flows, "all_paths": all_paths,
        "od_path_groups": od_path_groups, "od_pairs": OD_PAIRS,
    }


def plot(results, save_path="results/aberdeen_network.png"):
    """Plot realistic Aberdeen network with flows and saturation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    G = results["graph"]
    pos = {n: data["pos"] for n, data in G.nodes(data=True)}
    labels = {n: data["label"] for n, data in G.nodes(data=True)}

    # Flow map
    edge_flows = [G[u][v]["flow"] for u, v in G.edges()]
    max_flow = max(edge_flows) if edge_flows else 1
    edge_widths = [1 + 5 * f / max_flow for f in edge_flows]
    saturations = [G[u][v]["saturation"] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=300, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, labels={n: n for n in G.nodes()},
                            ax=ax1, font_size=8)
    edges = nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_widths,
                                   edge_color=saturations, edge_cmap=plt.cm.RdYlGn_r,
                                   edge_vmin=0, edge_vmax=1.5,
                                   connectionstyle="arc3,rad=0.1")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1.5))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, label="Saturation (flow/capacity)")
    ax1.set_title("Aberdeen Network: Link Saturation")

    # Add road name legend
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        ax1.annotate(data["label"], (x, y), fontsize=5,
                     xytext=(5, 5), textcoords="offset points", alpha=0.7)
    ax1.axis("off")

    # Bar chart of most congested links
    edge_data = [(G[u][v]["road"], G[u][v]["saturation"], G[u][v]["flow"])
                 for u, v in G.edges()]
    edge_data.sort(key=lambda x: x[1], reverse=True)
    top = edge_data[:10]

    roads = [d[0][:20] for d in top]
    sats = [d[1] for d in top]
    colors = [plt.cm.RdYlGn_r(s / 1.5) for s in sats]
    ax2.barh(range(len(roads)), sats, color=colors)
    ax2.set_yticks(range(len(roads)))
    ax2.set_yticklabels(roads, fontsize=8)
    ax2.set_xlabel("Saturation (flow/capacity)")
    ax2.set_title("Top 10 Most Congested Links")
    ax2.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="Capacity")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
