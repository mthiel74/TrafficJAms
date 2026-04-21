(* ::Package:: *)

(* AberdeenFullCity.wl -- full-city multi-agent simulation with traffic
   signals, roundabout yield, road-hierarchy routing, staggered spawn,
   and full IDM. Port of trafficjams/aberdeen_full_multiagent.py.

   Reuses IntersectionControl.wl for signal and roundabout logic. *)

BeginPackage["AberdeenFullCity`", {"IntersectionControl`"}];

LoadFullAberdeen::usage = "LoadFullAberdeen[path] parses an Overpass JSON dump into a driveable graph with highway/junction metadata needed for signal and roundabout control.";
SimulateFullAberdeen::usage = "SimulateFullAberdeen[net, opts] runs the 800-vehicle v2 simulation: full IDM, noisy hierarchy-weighted routing, staggered spawn, signal+roundabout control, probabilistic detour.";
RenderFullAberdeenFrame::usage = "RenderFullAberdeenFrame[sim, i] renders frame i.";
PlotFullAberdeen::usage = "PlotFullAberdeen[sim, path] exports the final frame; PlotFullAberdeen[sim] returns a Graphics.";
AnimateFullAberdeen::usage = "AnimateFullAberdeen[sim, path, opts] exports an animated GIF.";

Begin["`Private`"];

(* -------- Loader -------- *)

$DrivableHighways = {
  "motorway", "trunk", "primary", "secondary", "tertiary",
  "unclassified", "residential", "motorway_link", "trunk_link",
  "primary_link", "secondary_link", "tertiary_link", "living_street",
  "service", "road"
};

$DefaultFullCachePath =
  FileNameJoin[{ParentDirectory[DirectoryName[$InputFileName]],
     "cache", "4377299f6cd0a84ddd42b02725d1608d212e010c.json"}];

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
        Min[If[StringContainsQ[tag, "kph" | "km/h"], s/3.6, s 0.44704], 20.0],
        13.4
      ],
    True, 13.4
  ]
];

$HighwayCostFactor = <|
  "motorway" -> 0.5, "motorway_link" -> 0.6,
  "trunk" -> 0.6, "trunk_link" -> 0.7,
  "primary" -> 0.7, "primary_link" -> 0.8,
  "secondary" -> 0.85, "secondary_link" -> 0.9,
  "tertiary" -> 0.95, "tertiary_link" -> 1.0,
  "residential" -> 1.2, "service" -> 1.4,
  "living_street" -> 1.5, "unclassified" -> 1.1, "road" -> 1.1
|>;

LoadFullAberdeen[] := LoadFullAberdeen[$DefaultFullCachePath];

LoadFullAberdeen[path_String] := Module[
  {raw, elements, nodesList, waysList, nodeCoord, nodeTag, drivableWays,
   refCount, i, way, refs, tags, highway, junction, oneway, curr, segNodes,
   segLen, fromId, toId, usedNodes, edges, edgeLengths, edgeSpeeds,
   edgeHighway, edgeJunction, edgeCostFactor, keptNodeIds, centreLat,
   cosLat, nodePos, cx, cy, edgeList, gW, gU, scc, keepSet,
   keptEdgesMask, nodeHighway, nodeDegree, incomingEdgesAt, k},

  raw = Import[path, "RawJSON"];
  elements = raw["elements"];
  nodesList = Select[elements, #["type"] === "node" &];
  waysList = Select[elements, #["type"] === "way" &];

  nodeCoord = AssociationThread[
    (#["id"] & /@ nodesList) -> ({#["lat"], #["lon"]} & /@ nodesList)
  ];
  nodeTag = AssociationThread[
    (#["id"] & /@ nodesList) ->
      (Lookup[Lookup[#, "tags", <||>], "highway", ""] & /@ nodesList)
  ];

  drivableWays = Select[waysList,
    MemberQ[$DrivableHighways, Lookup[Lookup[#, "tags", <||>], "highway", ""]] &
  ];

  refCount = <||>;
  Do[
    refs = way["nodes"];
    Do[refCount[r] = Lookup[refCount, r, 0] + 1, {r, refs}],
    {way, drivableWays}
  ];
  isIntersection[nodeId_] := refCount[nodeId] >= 2;

  edges = {}; edgeLengths = {}; edgeSpeeds = {}; edgeHighway = {};
  edgeJunction = {}; edgeCostFactor = {}; usedNodes = <||>;
  Do[
    refs = way["nodes"];
    tags = Lookup[way, "tags", <||>];
    highway = Lookup[tags, "highway", ""];
    junction = Lookup[tags, "junction", ""];
    oneway = Lookup[tags, "oneway", "no"] === "yes" ||
             MemberQ[{"motorway", "motorway_link"}, highway];
    i = 1;
    While[i < Length[refs],
      segNodes = {refs[[i]]};
      segLen = 0.0;
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
        AppendTo[edgeHighway, highway];
        AppendTo[edgeJunction, junction];
        AppendTo[edgeCostFactor, Lookup[$HighwayCostFactor, highway, 1.0]];
        usedNodes[fromId] = True; usedNodes[toId] = True;
        If[!oneway,
          AppendTo[edges, {toId, fromId}];
          AppendTo[edgeLengths, segLen];
          AppendTo[edgeSpeeds, maxspeedToMs[Lookup[tags, "maxspeed", Missing[]]]];
          AppendTo[edgeHighway, highway];
          AppendTo[edgeJunction, junction];
          AppendTo[edgeCostFactor, Lookup[$HighwayCostFactor, highway, 1.0]];
        ]
      ];
      i = curr;
    ],
    {way, drivableWays}
  ];

  keptNodeIds = Keys[usedNodes];
  centreLat = Mean[(nodeCoord[#][[1]]) & /@ keptNodeIds];
  cosLat = Cos[centreLat Degree];
  nodePos = AssociationMap[
    Module[{lat, lon}, {lat, lon} = nodeCoord[#];
      {lon 111320.0 cosLat, lat 110540.0}] &,
    keptNodeIds
  ];
  cx = Mean[#[[1]] & /@ Values[nodePos]];
  cy = Mean[#[[2]] & /@ Values[nodePos]];
  nodePos = Association @ KeyValueMap[#1 -> ({#2[[1]] - cx, #2[[2]] - cy}) &, nodePos];

  edgeList = DirectedEdge[#[[1]], #[[2]]] & /@ edges;
  gU = Graph[keptNodeIds, edgeList];
  gW = Graph[keptNodeIds, edgeList, EdgeWeight -> edgeLengths];

  (* Restrict to largest strongly-connected component *)
  scc = First @ SortBy[ConnectedComponents[gW], -Length[#] &];
  keepSet = AssociationThread[scc -> True];
  keptEdgesMask = MapThread[
    (KeyExistsQ[keepSet, #1[[1]]] && KeyExistsQ[keepSet, #1[[2]]]) &,
    {edges}
  ];
  (* Down-filter *)
  edges = Pick[edges, keptEdgesMask];
  edgeLengths = Pick[edgeLengths, keptEdgesMask];
  edgeSpeeds = Pick[edgeSpeeds, keptEdgesMask];
  edgeHighway = Pick[edgeHighway, keptEdgesMask];
  edgeJunction = Pick[edgeJunction, keptEdgesMask];
  edgeCostFactor = Pick[edgeCostFactor, keptEdgesMask];
  keptNodeIds = scc;
  nodePos = KeySelect[nodePos, KeyExistsQ[keepSet, #] &];
  nodeHighway = KeySelect[nodeTag, KeyExistsQ[keepSet, #] &];

  edgeList = DirectedEdge[#[[1]], #[[2]]] & /@ edges;
  gU = Graph[keptNodeIds, edgeList];
  gW = Graph[keptNodeIds, edgeList, EdgeWeight -> edgeLengths];

  (* node degree & incoming edges per node *)
  nodeDegree = <||>; incomingEdgesAt = <||>;
  Do[nodeDegree[n] = 0, {n, keptNodeIds}];
  Do[
    nodeDegree[edges[[k, 1]]]++;
    nodeDegree[edges[[k, 2]]]++;
    incomingEdgesAt[edges[[k, 2]]] =
      Append[Lookup[incomingEdgesAt, Key[edges[[k, 2]]], {}], edges[[k]]],
    {k, Length[edges]}
  ];

  <|
    "graph" -> gW,
    "graphU" -> gU,
    "nodePos" -> nodePos,
    "nodeIds" -> keptNodeIds,
    "edges" -> edges,
    "edgeLength" -> edgeLengths,
    "edgeSpeed" -> edgeSpeeds,
    "edgeHighway" -> edgeHighway,
    "edgeJunction" -> edgeJunction,
    "edgeCostFactor" -> edgeCostFactor,
    "edgeIndex" -> AssociationThread[edges -> Range[Length[edges]]],
    "nodeDegree" -> nodeDegree,
    "nodeHighway" -> nodeHighway,
    "incomingEdgesAt" -> incomingEdgesAt,
    "bounds" -> {
      {Min[#[[1]] & /@ Values[nodePos]], Max[#[[1]] & /@ Values[nodePos]]},
      {Min[#[[2]] & /@ Values[nodePos]], Max[#[[2]] & /@ Values[nodePos]]}
    }
  |>
];

(* -------- Simulation -------- *)

Options[SimulateFullAberdeen] = {
  "nVehicles" -> 400,
  "dt" -> 1.0,
  "T" -> 300.0,
  "nFrames" -> 100,
  "seed" -> 123,
  "cycleTime" -> 60.0,
  "spawnWindow" -> 60.0,
  "detourProb" -> 0.1,
  "signalPenaltyDistance" -> 15.0
};

SimulateFullAberdeen[net_Association, OptionsPattern[]] := Module[
  {nVeh, dt, T, nFrames, seed, cycle, spawnWindow, detourProb, penaltyDist,
   nodeIds, edges, edgeLen, edgeSpeed, edgeIdx, edgeCost, edgeJunc, edgeHwy,
   gW, nodePos, nodeDegrees, nodeDegArr, nodeWeights, nodeClass, signalCtrls,
   incomingEdgesAt, vehicles, vi, veh, simTime, spf, step, sub, frames,
   edgeOcc, posList, spdList, edgeSpeedAccum, recordFrame, pickPath,
   pickNoisyPath, key, occ, gap, dv, u, v, e, elen, sl, distToNode, vClass,
   ctrl, sStar, effGap, acc, aMax, bComf, s0, THw, delta, newPath, origin,
   destination, attempts, jEdge, jCount, neighbours, detourNext, tail,
   noisyWeights, lastNode, k, ek, mean, nodeList, totalDeg, okEnter,
   endReached, newWAssoc, nodeIdxMap, lastSamp, canPick, keepGoing, gHier},

  nVeh = OptionValue["nVehicles"];
  dt = OptionValue["dt"];
  T = OptionValue["T"];
  nFrames = OptionValue["nFrames"];
  seed = OptionValue["seed"];
  cycle = OptionValue["cycleTime"];
  spawnWindow = OptionValue["spawnWindow"];
  detourProb = OptionValue["detourProb"];
  penaltyDist = OptionValue["signalPenaltyDistance"];

  SeedRandom[seed];

  nodeIds = net["nodeIds"];
  edges = net["edges"];
  edgeLen = net["edgeLength"];
  edgeSpeed = net["edgeSpeed"];
  edgeIdx = net["edgeIndex"];
  edgeCost = net["edgeCostFactor"];
  edgeJunc = net["edgeJunction"];
  edgeHwy = net["edgeHighway"];
  gW = net["graph"];
  nodePos = net["nodePos"];
  nodeDegrees = net["nodeDegree"];
  incomingEdgesAt = net["incomingEdgesAt"];

  (* Node classification: OSM node tag -> signal; any node on a
     junction=roundabout edge -> roundabout; otherwise uncontrolled. *)
  nodeClass = <||>;
  Do[
    nodeClass[n] = Switch[Lookup[net["nodeHighway"], n, ""],
      "traffic_signals", "signal",
      "mini_roundabout", "roundabout",
      "stop", "stop",
      "give_way", "giveWay",
      "yield", "giveWay",
      _, "uncontrolled"
    ],
    {n, nodeIds}
  ];
  Do[
    If[edgeJunc[[k]] === "roundabout",
      nodeClass[edges[[k, 1]]] = "roundabout";
      nodeClass[edges[[k, 2]]] = "roundabout";
    ],
    {k, Length[edges]}
  ];

  (* Signal controllers *)
  signalCtrls = BuildSignalControllers[incomingEdgesAt, nodeClass, cycle, seed];

  (* Degree-weighted OD sampling *)
  nodeList = nodeIds;
  nodeDegArr = Lookup[nodeDegrees, nodeList];
  totalDeg = Total[nodeDegArr];
  nodeWeights = nodeDegArr/totalDeg;

  (* Precompute a hierarchy-weighted graph once; picking paths is then
     a single FindShortestPath call on this reusable Graph. To approximate
     the Python version's noisy routing we multiply hierarchy-weighted
     edge lengths by a per-edge random 0.8..1.5 factor on this ONE graph,
     giving hierarchy preference plus frozen route diversity. *)
  Module[{weightedLens = MapThread[
      #1 Lookup[$HighwayCostFactor, #2, 1.0] RandomReal[{0.8, 1.5}] &,
      {edgeLen, edgeHwy}]},
    gHier = Graph[nodeIds, DirectedEdge[#[[1]], #[[2]]] & /@ edges,
                  EdgeWeight -> weightedLens]
  ];

  pickNoisyPath[o_, d_] := Quiet @ FindShortestPath[gHier, o, d];

  pickPath[] := Module[{o, d, p, attempt = 0},
    While[attempt < 20,
      attempt++;
      o = RandomChoice[nodeWeights -> nodeList];
      d = RandomChoice[nodeWeights -> nodeList];
      If[o === d, Continue[]];
      p = pickNoisyPath[o, d];
      If[ListQ[p] && Length[p] >= 3, Return[p, Module]];
    ];
    {}
  ];

  (* Build vehicles *)
  vehicles = Table[
    With[{p = pickPath[]},
      If[p === {},
        <|"active" -> False, "path" -> {}, "edgeIdx" -> 1, "posOnEdge" -> 0.,
          "speed" -> 0., "spawnTime" -> 0., "spawned" -> False|>,
        <|"active" -> True, "path" -> p, "edgeIdx" -> 1,
          "posOnEdge" -> 0., "speed" -> 0.,
          "spawnTime" -> RandomReal[{0, spawnWindow}], "spawned" -> False|>
      ]
    ],
    {nVeh}
  ];

  frames = {};
  simTime = 0.0;
  spf = Max[1, Round[T/dt/nFrames]];

  recordFrame[] := Module[
    {posList2 = {}, spdList2 = {}, edgeSpeedAccum2 = <||>, vi2, veh2,
     u2, v2, elen2, frac2, x02, y02, x12, y12, ek2, meanSpeeds},
    Do[
      veh2 = vehicles[[vi2]];
      If[veh2["active"] && veh2["spawned"] && veh2["edgeIdx"] < Length[veh2["path"]],
        u2 = veh2["path"][[veh2["edgeIdx"]]];
        v2 = veh2["path"][[veh2["edgeIdx"] + 1]];
        If[KeyExistsQ[edgeIdx, {u2, v2}],
          elen2 = edgeLen[[edgeIdx[{u2, v2}]]];
          frac2 = Min[veh2["posOnEdge"]/Max[elen2, 1.0], 1.0];
          {x02, y02} = nodePos[u2]; {x12, y12} = nodePos[v2];
          AppendTo[posList2,
            {x02 + frac2 (x12 - x02), y02 + frac2 (y12 - y02)}];
          AppendTo[spdList2, veh2["speed"]];
          ek2 = {u2, v2};
          edgeSpeedAccum2[ek2] = Append[Lookup[edgeSpeedAccum2, Key[ek2], {}], veh2["speed"]];
        ]
      ],
      {vi2, nVeh}
    ];
    meanSpeeds = Association @ KeyValueMap[#1 -> Mean[#2] &, edgeSpeedAccum2];
    AppendTo[frames, <|
      "positions" -> posList2, "speeds" -> spdList2,
      "edgeSpeeds" -> meanSpeeds, "simTime" -> simTime
    |>];
  ];

  recordFrame[];

  aMax = 2.0; bComf = 3.0; s0 = 2.0; THw = 1.2; delta = 4;

  Do[
    Do[
      simTime += dt;

      (* activate spawn-due vehicles *)
      Do[
        veh = vehicles[[vi]];
        If[veh["active"] && !veh["spawned"] && simTime >= veh["spawnTime"],
          veh["spawned"] = True;
          vehicles[[vi]] = veh;
        ],
        {vi, nVeh}
      ];

      (* per-edge occupancy *)
      edgeOcc = <||>;
      Do[
        veh = vehicles[[vi]];
        If[veh["active"] && veh["spawned"] && veh["edgeIdx"] < Length[veh["path"]],
          key = {veh["path"][[veh["edgeIdx"]]], veh["path"][[veh["edgeIdx"] + 1]]};
          edgeOcc[key] = Append[Lookup[edgeOcc, Key[key], {}],
            {veh["posOnEdge"], vi}];
        ],
        {vi, nVeh}
      ];
      edgeOcc = Map[SortBy[First], edgeOcc];

      Do[
        veh = vehicles[[vi]];
        If[!veh["active"] || !veh["spawned"] || veh["edgeIdx"] >= Length[veh["path"]], Continue[]];
        u = veh["path"][[veh["edgeIdx"]]];
        v = veh["path"][[veh["edgeIdx"] + 1]];
        If[!KeyExistsQ[edgeIdx, {u, v}], veh["active"] = False; vehicles[[vi]] = veh; Continue[]];
        e = edgeIdx[{u, v}];
        elen = edgeLen[[e]];
        sl = edgeSpeed[[e]];

        (* find leader *)
        occ = Lookup[edgeOcc, Key[{u, v}], {}];
        gap = elen; dv = 0.0;
        Do[
          If[occ[[k, 1]] > veh["posOnEdge"] + 0.1,
            gap = occ[[k, 1]] - veh["posOnEdge"];
            dv = veh["speed"] - vehicles[[occ[[k, 2]]]]["speed"];
            Break[];
          ],
          {k, Length[occ]}
        ];

        (* intersection control near stop line *)
        distToNode = elen - veh["posOnEdge"];
        If[0 < distToNode < penaltyDist,
          vClass = nodeClass[v];
          Which[
            vClass === "signal" && KeyExistsQ[signalCtrls, v],
              ctrl = signalCtrls[v];
              If[!SignalIsGreen[ctrl, {u, v}, simTime],
                sl = Min[sl, Max[distToNode 0.5, 0.5]]
              ],
            vClass === "roundabout",
              okEnter = True;
              (* simple yield: if any same-target edge has stopped leader near *)
              Do[
                If[ek === {u, v}, Continue[]];
                If[ek[[2]] === v,
                  Do[
                    If[edgeOcc[ek][[k, 1]] > edgeLen[[edgeIdx[ek]]] - penaltyDist &&
                       vehicles[[edgeOcc[ek][[k, 2]]]]["speed"] > 0.5,
                      okEnter = False; Break[]
                    ],
                    {k, Length[edgeOcc[ek]]}
                  ]
                ],
                {ek, Keys[edgeOcc]}
              ];
              If[!okEnter,
                sl = Min[sl, Max[distToNode 0.5, 0.5]]
              ]
          ]
        ];

        (* full IDM *)
        sStar = s0 + Max[veh["speed"] THw + veh["speed"] dv/(2 Sqrt[aMax bComf]), 0.0];
        effGap = Max[gap, s0 + 0.5];
        acc = aMax (1 - (veh["speed"]/Max[sl, 0.1])^delta - (sStar/effGap)^2);
        veh["speed"] = Max[veh["speed"] + acc dt, 0.0];
        veh["speed"] = Min[veh["speed"], sl];
        veh["posOnEdge"] += veh["speed"] dt;

        If[veh["posOnEdge"] >= elen,
          veh["posOnEdge"] -= elen;
          veh["edgeIdx"] += 1;

          (* probabilistic detour *)
          If[veh["edgeIdx"] < Length[veh["path"]] && RandomReal[] < detourProb,
            Module[{cur, nbrs, detourNxt, dest2, newTail},
              cur = veh["path"][[veh["edgeIdx"]]];
              nbrs = VertexOutComponent[gHier, cur, 1];
              nbrs = DeleteCases[nbrs, cur];
              If[Length[nbrs] > 1,
                detourNxt = RandomChoice[nbrs];
                dest2 = Last[veh["path"]];
                newTail = Quiet @ FindShortestPath[gHier, detourNxt, dest2];
                If[ListQ[newTail] && Length[newTail] >= 2,
                  veh["path"] = Join[Take[veh["path"], veh["edgeIdx"] - 1],
                                     {cur}, newTail];
                ]
              ]
            ]
          ];

          (* re-route when we reached the end of the path *)
          If[veh["edgeIdx"] >= Length[veh["path"]],
            origin = Last[veh["path"]];
            newPath = {};
            Do[
              destination = RandomChoice[nodeWeights -> nodeList];
              If[destination =!= origin,
                newPath = pickNoisyPath[origin, destination];
                If[ListQ[newPath] && Length[newPath] >= 2, Break[], newPath = {}];
              ],
              {attempts, 10}
            ];
            If[newPath =!= {},
              veh["path"] = newPath;
              veh["edgeIdx"] = 1;
              veh["posOnEdge"] = 0.0,
              veh["active"] = False;
            ];
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
    "nFrames" -> nFrames,
    "simTimeEnd" -> simTime
  |>
];

(* -------- Rendering -------- *)

RenderFullAberdeenFrame[sim_Association, i_Integer] := Module[
  {net, frames, frame, nFrames, bounds, edgeLines, vehicleGraphics, xr, yr, ar,
   nodePos, edges, speeds, positions, vmax, edgeSpeeds, congEdges},

  net = sim["net"];
  frames = sim["frames"];
  nFrames = Length[frames];
  frame = frames[[Clip[i, {1, nFrames}]]];
  positions = frame["positions"];
  speeds = frame["speeds"];
  edgeSpeeds = frame["edgeSpeeds"];
  vmax = 20.0;

  nodePos = net["nodePos"];
  edges = net["edges"];

  (* draw edges; recolour those with known mean speed *)
  edgeLines = MapThread[
    Module[{spd, col, thick},
      spd = Lookup[edgeSpeeds, Key[#2], Missing[]];
      If[MissingQ[spd],
        {GrayLevel[0.8], Thickness[0.0012], Line[{nodePos[#2[[1]]], nodePos[#2[[2]]]}]},
        col = Blend[{RGBColor[0.85, 0.1, 0.1], RGBColor[1.0, 0.65, 0.2],
                     RGBColor[0.1, 0.7, 0.2]}, Min[spd/vmax, 1.0]];
        thick = 0.002;
        {col, Thickness[thick], Line[{nodePos[#2[[1]]], nodePos[#2[[2]]]}]}
      ]
    ] &,
    {Range[Length[edges]], edges}
  ];

  vehicleGraphics = MapThread[
    {Blend[{RGBColor[0.85, 0.1, 0.1], RGBColor[1.0, 0.85, 0.2],
            RGBColor[0.1, 0.7, 0.2]}, Min[#2/vmax, 1.0]],
     Disk[#1, 25]} &,
    {positions, speeds}
  ];

  {xr, yr} = net["bounds"];
  ar = (yr[[2]] - yr[[1]])/(xr[[2]] - xr[[1]]);

  Graphics[
    {
      edgeLines,
      {EdgeForm[GrayLevel[0.3]], vehicleGraphics}
    },
    PlotRange -> {xr, yr},
    AspectRatio -> ar,
    Background -> GrayLevel[0.97],
    PlotLabel -> Row[{"Aberdeen full city (v2): ", Length[positions],
                       " vehicles, frame ", i, "/", nFrames,
                       ", t=", NumberForm[frame["simTime"], {5, 1}], "s"}],
    ImageSize -> 650
  ]
];

PlotFullAberdeen[sim_Association] := RenderFullAberdeenFrame[sim, Length[sim["frames"]]];

PlotFullAberdeen[sim_Association, path_String] := Module[{g = PlotFullAberdeen[sim]},
  Export[path, g, ImageResolution -> 150]; path];

Options[AnimateFullAberdeen] = {"frameRate" -> 12, "stride" -> 3};

AnimateFullAberdeen[sim_Association, path_String, OptionsPattern[]] := Module[
  {idx, images, fr, stride},
  stride = OptionValue["stride"];
  fr = OptionValue["frameRate"];
  idx = Range[1, Length[sim["frames"]], stride];
  images = RenderFullAberdeenFrame[sim, #] & /@ idx;
  Export[path, images,
    "DisplayDurations" -> ConstantArray[1.0/fr, Length[images]]];
  path
];

End[];
EndPackage[];
