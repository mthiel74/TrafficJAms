"""Multi-agent traffic simulation covering all of Aberdeen.

Extends the city-centre simulation to the full Aberdeen road network
with significantly more vehicles (~800).  Uses OSMnx to fetch the
drivable network for the whole city.
"""

import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString


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

    # ------------------------------------------------------------------
    # Create vehicles with random origin-destination pairs
    # ------------------------------------------------------------------
    vehicles = []
    for _ in range(n_vehicles):
        veh = Vehicle()
        for _attempt in range(30):
            o, d = rng.choice(nodes, size=2, replace=False)
            try:
                path = nx.shortest_path(G, o, d, weight="length")
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

    sim_time = 0.0
    spf = max(1, int(T / dt / n_frames))  # steps per frame

    # ------------------------------------------------------------------
    # Recording helper
    # ------------------------------------------------------------------
    frames = []

    def _record():
        positions = []
        speeds = []
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
            else:
                x0, y0 = node_pos[u]
                x1, y1 = node_pos[v]
                x = x0 + frac * (x1 - x0)
                y = y0 + frac * (y1 - y0)
            positions.append((x, y))
            speeds.append(veh.speed)
        frames.append({"positions": positions, "speeds": speeds})

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

                # Find gap to vehicle ahead
                key = (u, v)
                occ = edge_vehicles.get(key, [])
                gap = elen
                for j, (pos_j, vj) in enumerate(occ):
                    if pos_j > veh.pos_on_edge + 0.1:
                        gap = pos_j - veh.pos_on_edge
                        break

                # IDM-like acceleration
                s0 = 3.0
                desired_gap = s0 + veh.speed * 1.0
                acc = 2.0 * (1.0 - (veh.speed / max(sl, 0.1)) ** 4
                             - (desired_gap / max(gap, 0.5)) ** 2)
                veh.speed = max(veh.speed + acc * dt, 0.0)
                veh.speed = min(veh.speed, sl)
                veh.pos_on_edge += veh.speed * dt

                # Advance to next edge
                if veh.pos_on_edge >= elen:
                    veh.pos_on_edge -= elen
                    veh.edge_idx += 1
                    if veh.edge_idx >= len(veh.path) - 1:
                        origin = veh.path[-1]
                        for _attempt in range(15):
                            dest = rng.choice(nodes)
                            if dest != origin:
                                try:
                                    new_path = nx.shortest_path(G, origin, dest, weight="length")
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
