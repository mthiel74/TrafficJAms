(* ::Package:: *)

(* AberdeenCityNetwork.wl -- Multi-agent simulation on the real Aberdeen
   city-centre road network (OpenStreetMap, 800 m radius around Marischal
   College). Port of trafficjams/aberdeen_multiagent.py.

   The Overpass API JSON dumps shipped in ../cache/ are loaded, filtered
   for drivable highways, split at intersection nodes, and turned into a
   Wolfram Graph. Vehicles are then routed with FindShortestPath and
   stepped forward with a simple IDM-like car-following rule, recording
   frame-by-frame positions for rendering.

   Usage:
     Get["AberdeenCityNetwork.wl"];
     net = AberdeenCityNetwork`LoadAberdeenNetwork[];
     res = AberdeenCityNetwork`SimulateAberdeenCity[net, "nVehicles" -> 150];
     AberdeenCityNetwork`PlotAberdeenCity[res, "results/aberdeen_city.png"]; *)

BeginPackage["AberdeenCityNetwork`"];

LoadAberdeenNetwork::usage = "LoadAberdeenNetwork[] (or LoadAberdeenNetwork[path]) parses an Overpass JSON dump into a drivable-road graph association with keys graph, nodePos, edgeLength, edgeSpeed, edges, nodeIds.";
SimulateAberdeenCity::usage = "SimulateAberdeenCity[net, opts] runs a multi-agent IDM-like simulation on the parsed network and returns a per-frame position/speed record.";
PlotAberdeenCity::usage = "PlotAberdeenCity[sim] renders the final frame (vehicles coloured by speed) on the road network.";

Begin["`Private`"];

(* -------- OSM parsing -------- *)

$DrivableHighways = {
  "motorway", "trunk", "primary", "secondary", "tertiary",
  "unclassified", "residential", "motorway_link", "trunk_link",
  "primary_link", "secondary_link", "tertiary_link", "living_street",
  "service", "road"
};

$DefaultCachePath =
  FileNameJoin[{ParentDirectory[DirectoryName[$InputFileName]],
     "cache", "bb6a8c88fb22a35044e0ced8434d936907f63409.json"}];

haversine[{lat1_, lon1_}, {lat2_, lon2_}] := Module[
  {R = 6371000.0, dLat, dLon, a, rLat1, rLat2},
  rLat1 = lat1 Degree; rLat2 = lat2 Degree;
  dLat = (lat2 - lat1) Degree; dLon = (lon2 - lon1) Degree;
  a = Sin[dLat/2]^2 + Cos[rLat1] Cos[rLat2] Sin[dLon/2]^2;
  2 R ArcSin[Sqrt[a]]
];

maxspeedToMs[tag_] := Module[{s},
  Which[
    MissingQ[tag] || tag === None, 13.4,
    ListQ[tag], maxspeedToMs[First[tag]],
    StringQ[tag],
      s = Quiet @ Check[ToExpression[First[StringSplit[tag, " "]]], $Failed];
      If[NumberQ[s] && s > 0,
        Min[If[StringContainsQ[tag, "kph"|"km/h"], s/3.6, s*0.44704], 20.0],
        13.4
      ],
    True, 13.4
  ]
];

LoadAberdeenNetwork[] := LoadAberdeenNetwork[$DefaultCachePath];

LoadAberdeenNetwork[path_String] := Module[
  {raw, elements, nodesList, waysList, nodeCoord, drivableWays, refCount,
   way, refs, tags, highway, ways, isIntersection, way2, segs, i, curr,
   edges, edgeLength, edgeSpeed, edgeName, oneway, fromId, toId, segNodes,
   segLen, centreLat, cosLat, x, y, edgeList, gU, gW,
   edgeLengths, edgeSpeeds, keptNodeIds, nodePos, usedNodes},

  raw = Import[path, "RawJSON"];
  elements = raw["elements"];
  nodesList = Select[elements, #["type"] === "node" &];
  waysList = Select[elements, #["type"] === "way" &];

  nodeCoord = AssociationThread[
    (#["id"] & /@ nodesList) -> ({#["lat"], #["lon"]} & /@ nodesList)
  ];

  drivableWays = Select[waysList,
    MemberQ[$DrivableHighways, Lookup[Lookup[#, "tags", <||>], "highway", ""]] &
  ];

  (* node-reference count across drivable ways *)
  refCount = <||>;
  Do[
    refs = way["nodes"];
    Do[refCount[r] = Lookup[refCount, r, 0] + 1, {r, refs}],
    {way, drivableWays}
  ];

  isIntersection[nodeId_] := refCount[nodeId] >= 2;

  (* Build edges by splitting each way at intersections. *)
  edges = {}; edgeLengths = {}; edgeSpeeds = {}; usedNodes = <||>;
  Do[
    refs = way["nodes"];
    tags = Lookup[way, "tags", <||>];
    highway = Lookup[tags, "highway", ""];
    oneway = Lookup[tags, "oneway", "no"] === "yes" ||
             MemberQ[{"motorway", "motorway_link"}, highway];
    i = 1;
    While[i < Length[refs],
      segNodes = {refs[[i]]};
      segLen = 0.0;
      (* walk until next intersection or end of way *)
      For[curr = i + 1, curr <= Length[refs], curr++,
        AppendTo[segNodes, refs[[curr]]];
        segLen += haversine[nodeCoord[refs[[curr - 1]]], nodeCoord[refs[[curr]]]];
        If[curr == Length[refs] || isIntersection[refs[[curr]]], Break[]];
      ];
      fromId = First[segNodes]; toId = Last[segNodes];
      If[fromId =!= toId && segLen > 0.5,
        AppendTo[edges, {fromId, toId}];
        AppendTo[edgeLengths, segLen];
        AppendTo[edgeSpeeds, maxspeedToMs[Lookup[tags, "maxspeed", Missing[]]]];
        usedNodes[fromId] = True; usedNodes[toId] = True;
        If[!oneway,
          AppendTo[edges, {toId, fromId}];
          AppendTo[edgeLengths, segLen];
          AppendTo[edgeSpeeds, maxspeedToMs[Lookup[tags, "maxspeed", Missing[]]]];
        ]
      ];
      i = curr;
    ],
    {way, drivableWays}
  ];

  keptNodeIds = Keys[usedNodes];
  (* Simple equirectangular projection to metres, centred on mean lat *)
  centreLat = Mean[(nodeCoord[#][[1]]) & /@ keptNodeIds];
  cosLat = Cos[centreLat Degree];
  nodePos = AssociationMap[
    Module[{lat, lon}, {lat, lon} = nodeCoord[#];
      {lon 111320.0 cosLat, lat 110540.0}] &,
    keptNodeIds
  ];
  (* re-centre at (0, 0) *)
  nodePos = Module[{cx = Mean[#[[1]] & /@ Values[nodePos]],
                    cy = Mean[#[[2]] & /@ Values[nodePos]]},
    Association @ KeyValueMap[#1 -> ({#2[[1]] - cx, #2[[2]] - cy}) &, nodePos]
  ];

  edgeList = DirectedEdge[#[[1]], #[[2]]] & /@ edges;
  gU = Graph[keptNodeIds, edgeList];
  gW = Graph[keptNodeIds, edgeList, EdgeWeight -> edgeLengths];

  <|
    "graph" -> gW,
    "graphU" -> gU,
    "nodePos" -> nodePos,
    "nodeIds" -> keptNodeIds,
    "edges" -> edges,
    "edgeLength" -> edgeLengths,
    "edgeSpeed" -> edgeSpeeds,
    "edgeIndex" -> AssociationThread[edges -> Range[Length[edges]]],
    "bounds" -> {
      {Min[#[[1]] & /@ Values[nodePos]], Max[#[[1]] & /@ Values[nodePos]]},
      {Min[#[[2]] & /@ Values[nodePos]], Max[#[[2]] & /@ Values[nodePos]]}
    }
  |>
];

(* -------- Multi-agent simulation -------- *)

Options[SimulateAberdeenCity] = {
  "nVehicles" -> 150,
  "dt" -> 1.0,
  "T" -> 300.0,
  "nFrames" -> 120,
  "seed" -> 42
};

SimulateAberdeenCity[net_Association, OptionsPattern[]] := Module[
  {nVeh, dt, T, nFrames, seed, nodeIds, edges, edgeLen, edgeSpeed, edgeIdx,
   gW, nodePos, vehicles, n, path, origin, destination, attempts, frames,
   spf, step, sub, edgeVehicles, key, occ, gap, vi, veh, u, v, e,
   elen, sl, desiredGap, acc, s0, newPath, positions, speeds, recordFrame,
   pickPath, advanceFrame, x0, y0, x1, y1, frac, pickRandomNode, record,
   mainComponent},

  nVeh = OptionValue["nVehicles"];
  dt = OptionValue["dt"];
  T = OptionValue["T"];
  nFrames = OptionValue["nFrames"];
  seed = OptionValue["seed"];

  SeedRandom[seed];

  nodeIds = net["nodeIds"];
  edges = net["edges"];
  edgeLen = net["edgeLength"];
  edgeSpeed = net["edgeSpeed"];
  edgeIdx = net["edgeIndex"];
  gW = net["graph"];
  nodePos = net["nodePos"];
  n = Length[nodeIds];

  (* restrict vehicles to the largest weakly-connected component *)
  mainComponent = First[SortBy[WeaklyConnectedComponents[gW], -Length[#] &]];

  pickPath[] := Module[{o, d, p, k = 0},
    While[k < 30,
      k++;
      o = RandomChoice[mainComponent];
      d = RandomChoice[mainComponent];
      If[o === d, Continue[]];
      p = Quiet @ FindShortestPath[gW, o, d];
      If[ListQ[p] && Length[p] >= 3, Return[p, Module]];
    ];
    {}
  ];

  vehicles = Table[
    With[{p = pickPath[]},
      If[p === {},
        <|"active" -> False, "path" -> {}, "edgeIdx" -> 1, "posOnEdge" -> 0.,
          "speed" -> 0.|>,
        <|"active" -> True, "path" -> p, "edgeIdx" -> 1,
          "posOnEdge" -> RandomReal[{0, 0.3}] * edgeLen[[edgeIdx[{p[[1]], p[[2]]}]]],
          "speed" -> 0.5 edgeSpeed[[edgeIdx[{p[[1]], p[[2]]}]]]|>
      ]
    ],
    {nVeh}
  ];

  frames = {};
  spf = Max[1, Round[T/dt/nFrames]];

  recordFrame[] := Module[{posList = {}, spdList = {}, veh2, u2, v2, elen2,
     frac2, x02, y02, x12, y12},
    Do[
      veh2 = vehicles[[vi]];
      If[veh2["active"] && veh2["edgeIdx"] < Length[veh2["path"]],
        u2 = veh2["path"][[veh2["edgeIdx"]]];
        v2 = veh2["path"][[veh2["edgeIdx"] + 1]];
        elen2 = edgeLen[[edgeIdx[{u2, v2}]]];
        frac2 = Min[veh2["posOnEdge"]/Max[elen2, 1.0], 1.0];
        {x02, y02} = nodePos[u2]; {x12, y12} = nodePos[v2];
        AppendTo[posList, {x02 + frac2 (x12 - x02), y02 + frac2 (y12 - y02)}];
        AppendTo[spdList, veh2["speed"]];
      ],
      {vi, nVeh}
    ];
    AppendTo[frames, <|"positions" -> posList, "speeds" -> spdList|>];
  ];

  recordFrame[];

  Do[
    Do[
      (* occupancy per edge *)
      edgeVehicles = <||>;
      Do[
        veh = vehicles[[vi]];
        If[veh["active"] && veh["edgeIdx"] < Length[veh["path"]],
          key = {veh["path"][[veh["edgeIdx"]]], veh["path"][[veh["edgeIdx"] + 1]]};
          edgeVehicles[key] = Append[Lookup[edgeVehicles, Key[key], {}],
                                     {veh["posOnEdge"], vi}];
        ],
        {vi, nVeh}
      ];
      (* sort vehicles along each edge *)
      edgeVehicles = Map[SortBy[First], edgeVehicles];

      Do[
        veh = vehicles[[vi]];
        If[!veh["active"] || veh["edgeIdx"] >= Length[veh["path"]], Continue[]];
        u = veh["path"][[veh["edgeIdx"]]];
        v = veh["path"][[veh["edgeIdx"] + 1]];
        e = edgeIdx[{u, v}];
        elen = edgeLen[[e]];
        sl = edgeSpeed[[e]];

        (* find leader on same edge *)
        occ = Lookup[edgeVehicles, Key[{u, v}], {}];
        gap = elen;
        Do[
          If[occ[[j, 1]] > veh["posOnEdge"] + 0.1,
            gap = occ[[j, 1]] - veh["posOnEdge"];
            Break[];
          ],
          {j, Length[occ]}
        ];

        s0 = 3.0;
        desiredGap = s0 + veh["speed"] 1.0;
        acc = 2.0 (1.0 - (veh["speed"]/Max[sl, 0.1])^4 -
                  (desiredGap/Max[gap, 0.5])^2);
        veh["speed"] = Max[veh["speed"] + acc dt, 0.0];
        veh["speed"] = Min[veh["speed"], sl];
        veh["posOnEdge"] += veh["speed"] dt;

        If[veh["posOnEdge"] >= elen,
          veh["posOnEdge"] -= elen;
          veh["edgeIdx"] += 1;
          If[veh["edgeIdx"] >= Length[veh["path"]],
            (* re-route *)
            origin = Last[veh["path"]];
            newPath = {};
            Do[
              destination = RandomChoice[mainComponent];
              If[destination =!= origin,
                newPath = Quiet @ FindShortestPath[gW, origin, destination];
                If[ListQ[newPath] && Length[newPath] >= 2, Break[], newPath = {}];
              ],
              {attempts, 10}
            ];
            If[newPath =!= {},
              veh["path"] = newPath;
              veh["edgeIdx"] = 1;
              veh["posOnEdge"] = 0.0,
              veh["active"] = False;
            ]
          ]
        ];
        vehicles[[vi]] = veh,
        {vi, nVeh}
      ],
      {sub, spf}
    ];
    recordFrame[],
    {step, nFrames - 1}
  ];

  <|
    "frames" -> frames,
    "net" -> net,
    "nVehicles" -> nVeh,
    "nFrames" -> nFrames
  |>
];

(* -------- Rendering -------- *)

PlotAberdeenCity[sim_Association, OptionsPattern[]] := Module[
  {net, frames, frame, bounds, edgeLines, vehicleGraphics, xr, yr, ar,
   nodePos, edges, speeds, positions, vmax},

  net = sim["net"];
  frames = sim["frames"];
  frame = Last[frames];
  positions = frame["positions"];
  speeds = frame["speeds"];
  vmax = 20.0;

  nodePos = net["nodePos"];
  edges = net["edges"];

  edgeLines = Line[{nodePos[#[[1]]], nodePos[#[[2]]]}] & /@ edges;
  vehicleGraphics = MapThread[
    {Blend[{RGBColor[0.85, 0.1, 0.1], RGBColor[1.0, 0.85, 0.2],
            RGBColor[0.1, 0.7, 0.2]}, Min[#2/vmax, 1.0]],
     Disk[#1, 20]} &,
    {positions, speeds}
  ];

  {xr, yr} = net["bounds"];
  ar = (yr[[2]] - yr[[1]])/(xr[[2]] - xr[[1]]);

  Graphics[
    {
      {GrayLevel[0.75], Thickness[0.0015], edgeLines},
      {EdgeForm[Darker[Gray]], vehicleGraphics}
    },
    PlotRange -> {xr, yr},
    AspectRatio -> ar,
    Background -> GrayLevel[0.95],
    PlotLabel -> Row[{"Aberdeen city centre: ", Length[positions], " vehicles, frame ", Length[frames]}],
    ImageSize -> 900
  ]
];

PlotAberdeenCity[sim_Association, path_String] := Module[{g = PlotAberdeenCity[sim]},
  Export[path, g, ImageResolution -> 150]; path];

End[];
EndPackage[];
