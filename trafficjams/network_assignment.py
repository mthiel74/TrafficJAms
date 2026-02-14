"""Static traffic assignment with Beckmann formulation and BPR cost functions."""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import networkx as nx


def bpr_cost(flow, free_flow_time, capacity, alpha=0.15, beta=4.0):
    """BPR (Bureau of Public Roads) cost function."""
    return free_flow_time * (1 + alpha * (flow / capacity) ** beta)


def bpr_integral(flow, free_flow_time, capacity, alpha=0.15, beta=4.0):
    """Integral of BPR cost (for Beckmann objective)."""
    return free_flow_time * (flow + alpha * capacity / (beta + 1) *
                             (flow / capacity) ** (beta + 1))


def simulate():
    """Build Aberdeen-inspired network and solve traffic assignment.

    Returns dict with graph, flows, costs.
    """
    # Aberdeen-inspired network (simplified)
    # Nodes: key junctions
    nodes = {
        0: "Bridge of Don",
        1: "King St / A956",
        2: "Mounthooly",
        3: "Union St / Market St",
        4: "Bridge of Dee",
        5: "Anderson Drive North",
        6: "Anderson Drive South",
        7: "A90 North",
    }

    # Edges: (from, to, free_flow_time, capacity)
    edges = [
        (7, 0, 5, 2000),   # A90 to Bridge of Don
        (0, 1, 4, 1500),   # Bridge of Don to King St
        (1, 2, 3, 1200),   # King St to Mounthooly
        (2, 3, 2, 1000),   # Mounthooly to Union/Market
        (3, 4, 5, 1200),   # Union/Market to Bridge of Dee
        (7, 5, 6, 1800),   # A90 to Anderson Drive North
        (5, 6, 4, 1600),   # Anderson Drive N to S
        (6, 4, 3, 1400),   # Anderson Drive S to Bridge of Dee
        (5, 2, 3, 800),    # Anderson Drive N to Mounthooly
        (0, 5, 5, 900),    # Bridge of Don to Anderson Drive N
    ]

    n_edges = len(edges)
    fft = np.array([e[2] for e in edges], dtype=float)
    cap = np.array([e[3] for e in edges], dtype=float)

    # OD demand: from A90 North (7) to Bridge of Dee (4)
    # Two main routes: East (via King St) and West (via Anderson Drive)
    total_demand = 3000.0

    # Path-link incidence: enumerate simple paths
    # Path 1: 7->0->1->2->3->4 (edges 0,1,2,3,4)
    # Path 2: 7->5->6->4 (edges 5,6,7)
    # Path 3: 7->0->5->6->4 (edges 0,9,6,7)
    # Path 4: 7->5->2->3->4 (edges 5,8,3,4)
    paths = [
        [0, 1, 2, 3, 4],
        [5, 6, 7],
        [0, 9, 6, 7],
        [5, 8, 3, 4],
    ]
    n_paths = len(paths)

    # Path-link incidence matrix
    Delta = np.zeros((n_edges, n_paths))
    for p, path_edges in enumerate(paths):
        for e in path_edges:
            Delta[e, p] = 1

    def beckmann_objective(path_flows):
        link_flows = Delta @ path_flows
        return sum(bpr_integral(link_flows[e], fft[e], cap[e])
                   for e in range(n_edges))

    # Constraints: path flows sum to demand
    constraints = [{"type": "eq", "fun": lambda f: np.sum(f) - total_demand}]
    bounds = [(0, total_demand) for _ in range(n_paths)]
    x0 = np.ones(n_paths) * total_demand / n_paths

    result = minimize(beckmann_objective, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints)

    path_flows = result.x
    link_flows = Delta @ path_flows
    link_costs = np.array([bpr_cost(link_flows[e], fft[e], cap[e])
                           for e in range(n_edges)])

    # Path costs
    path_costs = Delta.T @ link_costs

    # Build networkx graph for visualisation
    G = nx.DiGraph()
    for idx, name in nodes.items():
        G.add_node(idx, label=name)
    for i, (u, v, t, c) in enumerate(edges):
        G.add_edge(u, v, flow=link_flows[i], cost=link_costs[i],
                   capacity=c, free_flow=t)

    return {"graph": G, "nodes": nodes, "edges": edges,
            "link_flows": link_flows, "link_costs": link_costs,
            "path_flows": path_flows, "path_costs": path_costs,
            "paths": paths, "total_demand": total_demand}


def plot(results, save_path="results/network_assignment.png"):
    """Plot network with flows."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    G = results["graph"]
    pos = {
        7: (0, 2), 0: (2, 3), 1: (3, 2.5), 2: (3, 1.5),
        3: (3, 0.5), 4: (2, -0.5), 5: (1, 2), 6: (1, 0.5),
    }

    labels = {n: results["nodes"][n] for n in G.nodes()}

    # Flow map
    edge_flows = [G[u][v]["flow"] for u, v in G.edges()]
    max_flow = max(edge_flows) if edge_flows else 1
    edge_widths = [1 + 4 * f / max_flow for f in edge_flows]
    edge_colors = [f / max_flow for f in edge_flows]

    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=500, node_color="lightblue")
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax1, font_size=7)
    edges_drawn = nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_widths,
                                         edge_color=edge_colors, edge_cmap=plt.cm.YlOrRd,
                                         connectionstyle="arc3,rad=0.1")
    ax1.set_title("Aberdeen Network: Equilibrium Flows")
    ax1.axis("off")

    # Path flow bar chart
    path_labels = [f"Path {i+1}" for i in range(len(results["path_flows"]))]
    bars = ax2.bar(path_labels, results["path_flows"], color="steelblue")
    ax2.set_ylabel("Flow (vehicles)")
    ax2.set_title("Path Flow Distribution (Wardrop Equilibrium)")

    # Annotate with path costs
    for bar, cost in zip(bars, results["path_costs"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 f"cost={cost:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path
