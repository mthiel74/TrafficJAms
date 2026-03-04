"""Multi-agent traffic simulation covering all of Aberdeen.

Extends the city-centre simulation to the full Aberdeen road network
with significantly more vehicles (~800).  Uses OSMnx to fetch the
drivable network for the whole city.
"""

import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString
from trafficjams.intersection_control import (
    classify_nodes, build_signal_controllers,
    roundabout_can_enter, intersection_virtual_gap,
)


def fetch_network():
    """Download the full Aberdeen drivable street network."""
    G = ox.graph_from_place(
        "Aberdeen, Scotland, UK",
        network_type="drive",
        simplify=True,
    )
    G = ox.project_graph(G)
    return G


def _edge_length(G, u, v, key=0):
    return G[u][v][key].get("length", 50.0)


def simulate(G=None, n_vehicles=800, T=400, dt=1.0, n_frames=200):
    """Run multi-agent simulation on the full Aberdeen network.

    Parameters
    ----------
    G : networkx.MultiDiGraph or None
        Pre-fetched OSM graph.  Fetched automatically if *None*.
    n_vehicles : int
        Number of vehicles to simulate (default 800).
    T : float
        Total simulation time in seconds.
    dt : float
        Integration time-step in seconds.
    n_frames : int
        Number of animation frames to record.

    Returns
    -------
    dict with keys: frames, graph, node_pos, n_vehicles, n_frames
    """
    if G is None:
        G = fetch_network()

    # Restrict to largest strongly connected component so all OD pairs are reachable
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_scc).copy()

    nodes = list(G.nodes())
    edges_data = {}
    edge_geoms = {}  # (u, v, k) -> Shapely LineString for interpolation
    for u, v, k, d in G.edges(data=True, keys=True):
        length = d.get("length", 50.0)
        maxspeed = d.get("maxspeed", "30")
        if isinstance(maxspeed, list):
            maxspeed = maxspeed[0]
        try:
            speed_limit = float(maxspeed) * 0.44704  # mph to m/s
        except (ValueError, TypeError):
            speed_limit = 13.4  # 30 mph default
        speed_limit = min(speed_limit, 20.0)
        edges_data[(u, v, k)] = {"length": length, "speed_limit": speed_limit}
        # Store geometry for position interpolation
        if "geometry" in d:
            edge_geoms[(u, v, k)] = d["geometry"]

    # Node positions (projected coordinates)
    node_pos = {}
    for n, d in G.nodes(data=True):
        node_pos[n] = (d["x"], d["y"])

    rng = np.random.default_rng(123)

    # Weight OD node selection by degree (major intersections attract more trips)
    node_degrees = np.array([G.degree(n) for n in nodes], dtype=float)
    node_weights = node_degrees / node_degrees.sum()

    # ------------------------------------------------------------------
    # Vehicle class
    # ------------------------------------------------------------------
    class Vehicle:
        __slots__ = ["path", "edge_idx", "pos_on_edge", "speed", "active",
                     "spawn_time", "spawned"]
        def __init__(self):
            self.path = []
            self.edge_idx = 0
            self.pos_on_edge = 0.0
            self.speed = 0.0
            self.active = True
            self.spawn_time = 0.0
            self.spawned = False

    # Road hierarchy multipliers: major roads get lower cost, encouraging use
    _highway_cost_factor = {
        "motorway": 0.5, "motorway_link": 0.6,
        "trunk": 0.6, "trunk_link": 0.7,
        "primary": 0.7, "primary_link": 0.8,
        "secondary": 0.85, "secondary_link": 0.9,
        "tertiary": 0.95, "tertiary_link": 1.0,
        "residential": 1.2, "service": 1.4,
        "living_street": 1.5, "unclassified": 1.1,
    }

    def _noisy_route(G, o, d, rng):
        """Shortest path with hierarchy-biased noisy weights for route diversity."""
        for u, v, k, data in G.edges(data=True, keys=True):
            hw = data.get("highway", "unclassified")
            if isinstance(hw, list):
                hw = hw[0]
            hf = _highway_cost_factor.get(hw, 1.0)
            data["noisy_length"] = data.get("length", 50.0) * hf * rng.uniform(0.8, 1.5)
        return nx.shortest_path(G, o, d, weight="noisy_length")

    # ------------------------------------------------------------------
    # Create vehicles with random origin-destination pairs
    # ------------------------------------------------------------------
    vehicles = []
    for _ in range(n_vehicles):
        veh = Vehicle()
        for _attempt in range(30):
            idx = rng.choice(len(nodes), size=2, replace=False, p=node_weights)
            o, d = nodes[idx[0]], nodes[idx[1]]
            try:
                path = _noisy_route(G, o, d, rng)
                if len(path) >= 3:
                    veh.path = path
                    break
            except nx.NetworkXNoPath:
                continue
        if not veh.path:
            veh.active = False
        else:
            veh.pos_on_edge = 0.0
            veh.speed = 0.0
            # Stagger entry: vehicles spawn over the first 60 seconds
            veh.spawn_time = rng.uniform(0, 60.0)
            veh.spawned = False
        vehicles.append(veh)

    # Intersection control setup
    node_class = classify_nodes(G)
    signal_controllers = build_signal_controllers(G, node_class, rng=rng)
    # Use unique neighbor count (not MultiDiGraph degree which inflates values)
    node_arm_count = {}
    for n in G.nodes():
        node_arm_count[n] = len(set(G.predecessors(n)) | set(G.successors(n)))

    sim_time = 0.0
    spf = max(1, int(T / dt / n_frames))  # steps per frame

    # ------------------------------------------------------------------
    # Recording helper
    # ------------------------------------------------------------------
    frames = []

    def _record():
        positions = []
        speeds = []
        headings = []  # for directional markers
        edge_speed_accum = {}  # (u,v) -> [speeds] for congestion colouring
        for veh in vehicles:
            if not veh.active or not veh.spawned or veh.edge_idx >= len(veh.path) - 1:
                continue
            u = veh.path[veh.edge_idx]
            v = veh.path[veh.edge_idx + 1]
            elen = _edge_length(G, u, v)
            frac = min(veh.pos_on_edge / max(elen, 1.0), 1.0)
            geom = edge_geoms.get((u, v, 0))
            if geom is not None:
                pt = geom.interpolate(frac, normalized=True)
                x, y = pt.x, pt.y
                # Compute heading from geometry tangent
                frac2 = min(frac + 0.01, 1.0)
                pt2 = geom.interpolate(frac2, normalized=True)
                dx, dy = pt2.x - pt.x, pt2.y - pt.y
            else:
                x0, y0 = node_pos[u]
                x1, y1 = node_pos[v]
                x = x0 + frac * (x1 - x0)
                y = y0 + frac * (y1 - y0)
                dx, dy = x1 - x0, y1 - y0
            heading = np.arctan2(dy, dx) if (dx != 0 or dy != 0) else 0.0
            # Point 19: Lateral lane offset for multi-lane feel
            ek = (u, v)
            lane_count = edge_speed_accum.get(ek, [])
            lane_idx = len(lane_count)  # current count before append
            offset_m = (lane_idx % 3 - 1) * 4.0  # -4, 0, +4 metres
            perp_x = -np.sin(heading) * offset_m
            perp_y = np.cos(heading) * offset_m
            positions.append((x + perp_x, y + perp_y))
            speeds.append(veh.speed)
            headings.append(heading)
            # Accumulate per-edge speeds
            ek = (u, v)
            if ek not in edge_speed_accum:
                edge_speed_accum[ek] = []
            edge_speed_accum[ek].append(veh.speed)
        # Compute mean speed per edge
        edge_mean_speeds = {}
        for ek, spds in edge_speed_accum.items():
            edge_mean_speeds[ek] = np.mean(spds)
        frames.append({
            "positions": positions,
            "speeds": speeds,
            "headings": headings,
            "edge_speeds": edge_mean_speeds,
        })

    _record()

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    for step in range(n_frames - 1):
        for _ in range(spf):
            sim_time += dt
            # Activate vehicles whose spawn time has arrived
            for veh in vehicles:
                if veh.active and not veh.spawned and sim_time >= veh.spawn_time:
                    veh.spawned = True

            # Build per-edge occupancy lists
            edge_vehicles = {}
            for vi, veh in enumerate(vehicles):
                if not veh.active or not veh.spawned or veh.edge_idx >= len(veh.path) - 1:
                    continue
                u = veh.path[veh.edge_idx]
                v = veh.path[veh.edge_idx + 1]
                key = (u, v)
                if key not in edge_vehicles:
                    edge_vehicles[key] = []
                edge_vehicles[key].append((veh.pos_on_edge, vi))

            for key in edge_vehicles:
                edge_vehicles[key].sort()

            # Update vehicles
            for vi, veh in enumerate(vehicles):
                if not veh.active or not veh.spawned or veh.edge_idx >= len(veh.path) - 1:
                    continue
                u = veh.path[veh.edge_idx]
                v = veh.path[veh.edge_idx + 1]
                e = (u, v, 0)
                elen = _edge_length(G, u, v)
                sl = edges_data.get(e, {"speed_limit": 10.0})["speed_limit"]

                # Find gap and speed of vehicle ahead (same edge)
                key = (u, v)
                occ = edge_vehicles.get(key, [])
                gap = elen - veh.pos_on_edge  # default: distance to end of edge
                dv = 0.0  # speed difference (self - leader)
                found_leader = False
                for j, (pos_j, vj_idx) in enumerate(occ):
                    if pos_j > veh.pos_on_edge + 0.1:
                        gap = pos_j - veh.pos_on_edge
                        dv = veh.speed - vehicles[vj_idx].speed
                        found_leader = True
                        break
                # Cross-edge gap: check first vehicle on the next edge
                if not found_leader and veh.edge_idx + 2 < len(veh.path):
                    next_u = veh.path[veh.edge_idx + 1]
                    next_v = veh.path[veh.edge_idx + 2]
                    next_occ = edge_vehicles.get((next_u, next_v), [])
                    if next_occ:
                        first_pos, first_idx = next_occ[0]
                        cross_gap = (elen - veh.pos_on_edge) + first_pos
                        if cross_gap < gap:
                            gap = cross_gap
                            dv = veh.speed - vehicles[first_idx].speed

                # Intersection control via speed-limit reduction only.
                # Never skip the IDM or edge-advance logic — that causes
                # permanent gridlock. Instead, reduce sl so the IDM naturally
                # decelerates, and restore sl when the constraint clears.
                dist_to_node = elen - veh.pos_on_edge
                LOOKAHEAD = 40.0
                if dist_to_node < LOOKAHEAD and dist_to_node > 0:
                    v_class = node_class.get(v, "uncontrolled")
                    if v_class == "signal":
                        ctrl = signal_controllers.get(v)
                        if ctrl and not ctrl.is_green((u, v), sim_time):
                            # Red light: reduce speed limit proportional to distance
                            sl = min(sl, max(dist_to_node * 0.15, 0.1))
                    elif v_class == "roundabout":
                        if not roundabout_can_enter(G, v, edge_vehicles, vehicles,
                                                    dist_to_node, veh.speed,
                                                    approach_edge=(u, v)):
                            sl = min(sl, max(dist_to_node * 0.15, 0.1))
                    elif v_class in ("stop", "give_way"):
                        arms = node_arm_count.get(v, 2)
                        if arms >= 4:
                            sl = min(sl, sl * 0.5)

                # Full IDM acceleration
                # Parameters: a=2.0 m/s², b=3.0 m/s², s0=2.0 m, T=1.2 s, delta=4
                a_max = 2.0    # max acceleration
                b_comf = 3.0   # comfortable deceleration
                s0 = 2.0       # minimum gap (reduced to prevent gridlock)
                T_hw = 1.2     # safe time headway
                delta = 4      # acceleration exponent
                s_star = s0 + max(veh.speed * T_hw + veh.speed * dv / (2.0 * (a_max * b_comf) ** 0.5), 0.0)
                # Ensure gap > s_star for stopped vehicles so they can restart
                effective_gap = max(gap, s0 + 0.5)
                acc = a_max * (1.0 - (veh.speed / max(sl, 0.1)) ** delta
                              - (s_star / effective_gap) ** 2)
                veh.speed = max(veh.speed + acc * dt, 0.0)
                veh.speed = min(veh.speed, sl)
                veh.pos_on_edge += veh.speed * dt

                # Advance to next edge
                if veh.pos_on_edge >= elen:
                    veh.pos_on_edge -= elen
                    veh.edge_idx += 1
                    # Probabilistic detour: 10% chance to deviate
                    if veh.edge_idx < len(veh.path) - 1 and rng.random() < 0.1:
                        current_node = veh.path[veh.edge_idx]
                        neighbours = list(G.successors(current_node))
                        if len(neighbours) > 1:
                            detour_next = neighbours[rng.integers(len(neighbours))]
                            dest = veh.path[-1]
                            try:
                                tail = nx.shortest_path(G, detour_next, dest, weight="length")
                                veh.path = list(veh.path[:veh.edge_idx]) + [current_node] + tail
                                # edge_idx stays the same (pointing at current_node)
                            except nx.NetworkXNoPath:
                                pass  # keep original path
                    if veh.edge_idx >= len(veh.path) - 1:
                        origin = veh.path[-1]
                        for _attempt in range(15):
                            dest = rng.choice(nodes)
                            if dest != origin:
                                try:
                                    new_path = _noisy_route(G, origin, dest, rng)
                                    if len(new_path) >= 2:
                                        veh.path = new_path
                                        veh.edge_idx = 0
                                        veh.pos_on_edge = 0.0
                                        break
                                except nx.NetworkXNoPath:
                                    continue
                        else:
                            veh.active = False

        _record()

        if (step + 1) % 50 == 0:
            n_active = sum(1 for v in vehicles if v.active and v.edge_idx < len(v.path) - 1)
            print(f"  frame {step+1}/{n_frames-1}  active={n_active}")

    return {
        "frames": frames,
        "graph": G,
        "node_pos": node_pos,
        "n_vehicles": n_vehicles,
        "n_frames": n_frames,
    }
