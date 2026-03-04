"""Intersection control: traffic signals, roundabouts, and uncontrolled delays.

Classifies OSMnx graph nodes by control type and provides models for
each type that integrate with the IDM car-following loop via virtual
gap distances.
"""

import numpy as np


def classify_nodes(G):
    """Classify every node as signal, roundabout, stop, give_way, or uncontrolled.

    Uses OSM node-level 'highway' tags and edge-level 'junction' tags.
    """
    roundabout_nodes = set()
    for u, v, data in G.edges(data=True):
        junc = data.get("junction", "")
        if isinstance(junc, list):
            junc = junc[0]
        if junc == "roundabout":
            roundabout_nodes.add(u)
            roundabout_nodes.add(v)

    node_class = {}
    for node, data in G.nodes(data=True):
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0]
        if hw == "traffic_signals":
            node_class[node] = "signal"
        elif hw == "mini_roundabout" or node in roundabout_nodes:
            node_class[node] = "roundabout"
        elif hw == "stop":
            node_class[node] = "stop"
        elif hw in ("give_way", "yield"):
            node_class[node] = "give_way"
        else:
            node_class[node] = "uncontrolled"
    return node_class


class TrafficSignalController:
    """Fixed-time round-robin signal controller for one node.

    Aberdeen defaults: 90s cycle, 4s amber/intergreen (UK standard).
    """

    def __init__(self, node_id, incoming_edges, cycle_time=90.0,
                 amber_time=4.0, offset=0.0):
        self.node_id = node_id
        self.cycle_time = cycle_time
        self.amber_time = amber_time
        self.offset = offset
        # Each incoming edge gets its own phase (conservative)
        self.phases = [set([e]) for e in incoming_edges] if len(incoming_edges) > 2 \
            else [set(incoming_edges)]
        n_phases = len(self.phases)
        lost = amber_time * n_phases
        usable = max(cycle_time - lost, 10.0 * n_phases)
        self.green_times = [usable / n_phases] * n_phases
        self._phase_starts = []
        t = 0.0
        for g in self.green_times:
            self._phase_starts.append(t)
            t += g + amber_time

    def is_green(self, edge, sim_time):
        """Return True if edge (u,v) has green at sim_time."""
        t = (sim_time - self.offset) % self.cycle_time
        for i, start in enumerate(self._phase_starts):
            end = start + self.green_times[i]
            if start <= t < end and edge in self.phases[i]:
                return True
        return False


def build_signal_controllers(G, node_class, cycle_time=90.0, rng=None):
    """Build one TrafficSignalController per signal node."""
    if rng is None:
        rng = np.random.default_rng(0)
    controllers = {}
    for node, cls in node_class.items():
        if cls != "signal":
            continue
        in_edges = [(u, node) for u in G.predecessors(node)]
        if not in_edges:
            continue
        offset = rng.uniform(0, cycle_time)
        controllers[node] = TrafficSignalController(
            node_id=node, incoming_edges=in_edges,
            cycle_time=cycle_time, offset=float(offset))
    return controllers


def roundabout_can_enter(G, node_id, edge_vehicles, vehicles, elen_remaining,
                         veh_speed, critical_headway=4.5):
    """Check if a vehicle can enter a roundabout (gap acceptance).

    Returns True if there is a sufficient gap in circulating traffic.
    UK critical headway ~4.5s (TRL Report 281).
    """
    # Check all vehicles on edges leading to this roundabout node
    for u in G.predecessors(node_id):
        occ = edge_vehicles.get((u, node_id), [])
        for pos_j, vj_idx in occ:
            leader = vehicles[vj_idx]
            if leader.speed < 0.5:
                continue
            elen = G[u][node_id][0].get("length", 50.0)
            dist_to_node = elen - pos_j
            time_to_arrive = dist_to_node / max(leader.speed, 0.1)
            if time_to_arrive < critical_headway:
                return False
    return True


def intersection_virtual_gap(node_degree, node_class):
    """Return a virtual gap distance (m) to model intersection delay.

    The IDM will naturally decelerate toward this gap, producing
    realistic delay without a separate timer.
    """
    base_delay = {"uncontrolled": 2.0, "give_way": 4.0, "stop": 8.0}.get(node_class, 2.0)
    degree_factor = 1.0 + 0.15 * max(node_degree - 3, 0)
    return max(base_delay * degree_factor, 3.0)
