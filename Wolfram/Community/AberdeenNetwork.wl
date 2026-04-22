(* ::Package:: *)

(* AberdeenNetwork.wl -- Realistic 15-node Aberdeen junction network with
   multi-OD Wardrop equilibrium assignment. Port of
   trafficjams/aberdeen_network.py. *)

BeginPackage["AberdeenNetwork`"];

AberdeenNodes::usage = "Association of node id -> {name, {x, y}}.";
AberdeenEdges::usage = "List of {from, to, freeFlowMin, capVehHr, roadName}.";
AberdeenOD::usage = "List of {origin, destination, demandVehHr} demand triples.";
SimulateAberdeenNetwork::usage = "SimulateAberdeenNetwork[opts] solves the Wardrop equilibrium on the Aberdeen network.";
PlotAberdeenNetwork::usage = "PlotAberdeenNetwork[res] draws saturation map + top-10 congestion chart.";

Begin["`Private`"];

AberdeenNodes = <|
  0  -> {"Bridge of Don Roundabout",   {1.2, 4.0}},
  1  -> {"Haudagain Roundabout",       {0.5, 3.2}},
  2  -> {"Persley Bridge",             {0.2, 3.5}},
  3  -> {"King St / St Machar Dr",     {1.4, 3.2}},
  4  -> {"Mounthooly Roundabout",      {1.2, 2.5}},
  5  -> {"Berryden / Westburn",        {0.7, 2.5}},
  6  -> {"Union St / King St",         {1.1, 1.8}},
  7  -> {"Union St / Market St",       {0.9, 1.6}},
  8  -> {"Guild St / Bridge St",       {0.8, 1.3}},
  9  -> {"Anderson Drive / A90",       {0.0, 2.0}},
  10 -> {"Anderson Drive / Queen's Rd",{0.0, 1.2}},
  11 -> {"Bridge of Dee",              {0.5, 0.3}},
  12 -> {"A90 North (AWPR Junction)",  {0.5, 4.5}},
  13 -> {"A96 / Auchmill Rd",          {0.0, 3.8}},
  14 -> {"Beach Boulevard / Esplanade",{1.8, 2.8}}
|>;

AberdeenEdges = {
  {12, 0,  3, 3000, "A90 Parkway"},
  {12, 13, 4, 2500, "A90/A96 Link"},
  {13, 1,  3, 2000, "A96 Auchmill Rd"},
  {13, 2,  2, 1500, "Mugiemoss Rd"},
  {0,  3,  4, 1800, "King Street (north)"},
  {0,  1,  5, 1200, "Tillydrone Ave"},
  {0,  14, 3, 1000, "Beach Boulevard"},
  {1,  2,  2, 1200, "Persley Rd"},
  {1,  5,  3, 1400, "Berryden Rd"},
  {2,  9,  4, 1600, "Anderson Dr (north)"},
  {3,  4,  3, 1200, "King Street (mid)"},
  {3,  14, 2, 800,  "Linksfield Rd"},
  {4,  6,  3, 1000, "King Street (south)"},
  {4,  5,  2, 900,  "Loch St / George St"},
  {5,  6,  3, 1000, "Skene St / Rosemount"},
  {5,  9,  3, 1400, "Westburn Rd"},
  {6,  7,  2, 1200, "Union Street"},
  {7,  8,  2, 1100, "Market Street"},
  {8,  11, 5, 1500, "Holburn St / Gt Southern Rd"},
  {9,  10, 4, 2000, "Anderson Drive (south)"},
  {10, 11, 3, 1800, "Anderson Dr / Garthdee"},
  {10, 7,  4, 1000, "Queen's Rd / Albyn Pl"},
  {14, 4,  3, 800,  "Gallowgate"}
};

AberdeenOD = {
  {12, 11, 4000},  (* A90 North to Bridge of Dee -- north-south through *)
  {12, 7,  2000},  (* A90 North to city centre *)
  {13, 7,  1500},  (* A96 to city centre *)
  {13, 11, 1000}   (* A96 to Bridge of Dee *)
};

bprCost[f_, fft_, cap_, alpha_: 0.15, beta_: 4.0] :=
  fft (1 + alpha (f/cap)^beta);

bprIntegral[f_, fft_, cap_, alpha_: 0.15, beta_: 4.0] :=
  fft (f + alpha cap/(beta + 1) (f/cap)^(beta + 1));

(* k-shortest simple paths by free-flow weight. Enumerates all simple paths
   on an unweighted copy (bounded by maxHops), then ranks by the weighted
   graph's free-flow total cost. *)
kShortestPaths[gW_Graph, gU_Graph, weightAssoc_Association,
  o_, d_, k_Integer, maxHops_Integer] := Module[{allPaths, edgeCost, costs, idx},
  allPaths = FindPath[gU, o, d, maxHops, All];
  If[allPaths === {}, Return[{}]];
  edgeCost[u_, v_] := weightAssoc[{u, v}];
  costs = Total[MapThread[edgeCost, {Most[#], Rest[#]}]] & /@ allPaths;
  idx = Ordering[costs];
  Take[allPaths[[idx]], UpTo[k]]
];

Options[SimulateAberdeenNetwork] = {"maxPaths" -> 5, "maxHops" -> 10};

SimulateAberdeenNetwork[OptionsPattern[]] := Module[
  {edges, fft, cap, roadNames, nEdges, gW, gU, weightAssoc, edgeIndex,
   maxPaths, maxHops, allPaths, odGroups, odIdx, orig, dest, demand,
   nodePaths, nodePath, edgeIndices, startIdx, delta, vars, linkFlows,
   objective, constraints, sol, pathFlows, linkCosts, pathCosts,
   saturation, result, e},

  maxPaths = OptionValue["maxPaths"]; maxHops = OptionValue["maxHops"];

  edges = AberdeenEdges;
  nEdges = Length[edges];
  fft = N @ edges[[All, 3]];
  cap = N @ edges[[All, 4]];
  roadNames = edges[[All, 5]];

  (* Weighted and unweighted directed graphs: unweighted used for path
     enumeration (hop-count bound), weighted for ranking cost. *)
  gW = Graph[
    Keys[AberdeenNodes],
    DirectedEdge[#[[1]], #[[2]]] & /@ edges,
    EdgeWeight -> fft
  ];
  gU = Graph[
    Keys[AberdeenNodes],
    DirectedEdge[#[[1]], #[[2]]] & /@ edges
  ];
  weightAssoc = AssociationThread[
    ({#[[1]], #[[2]]} & /@ edges) -> fft
  ];

  (* edge index lookup *)
  edgeIndex = AssociationThread[
    ({#[[1]], #[[2]]} & /@ edges) -> Range[nEdges]
  ];

  (* Enumerate paths for each OD pair *)
  allPaths = {};
  odGroups = {};
  Do[
    {orig, dest, demand} = AberdeenOD[[odIdx]];
    nodePaths = kShortestPaths[gW, gU, weightAssoc, orig, dest, maxPaths, maxHops];
    startIdx = Length[allPaths] + 1;
    Do[
      edgeIndices = Table[
        edgeIndex[{nodePath[[i]], nodePath[[i + 1]]}],
        {i, Length[nodePath] - 1}
      ];
      AppendTo[allPaths, <|"od" -> odIdx, "nodes" -> nodePath,
                          "edges" -> edgeIndices|>],
      {nodePath, nodePaths}
    ];
    AppendTo[odGroups, {startIdx, Length[allPaths]}],
    {odIdx, Length[AberdeenOD]}
  ];

  (* Path-link incidence *)
  delta = ConstantArray[0, {nEdges, Length[allPaths]}];
  Do[
    Do[delta[[e, p]] = 1, {e, allPaths[[p, "edges"]]}],
    {p, Length[allPaths]}
  ];

  vars = Array[f, Length[allPaths]];
  linkFlows = delta . vars;
  objective = Sum[
    bprIntegral[linkFlows[[e]], fft[[e]], cap[[e]]],
    {e, nEdges}
  ];

  constraints = Join[
    Table[
      Total[vars[[odGroups[[odIdx, 1]];;odGroups[[odIdx, 2]]]]] ==
        AberdeenOD[[odIdx, 3]],
      {odIdx, Length[AberdeenOD]}
    ],
    Thread[vars >= 0]
  ];

  sol = FindMinimum[
    {objective, constraints}, vars,
    Method -> "InteriorPoint", MaxIterations -> 2000
  ];

  pathFlows = Clip[vars /. sol[[2]], {0, Infinity}];
  linkFlows = delta . pathFlows;
  linkCosts = MapThread[bprCost[#1, #2, #3] &, {linkFlows, fft, cap}];
  pathCosts = Transpose[delta] . linkCosts;
  saturation = linkFlows/cap;

  <|
    "graph" -> gW, "edges" -> edges, "roadNames" -> roadNames,
    "linkFlows" -> linkFlows, "linkCosts" -> linkCosts,
    "saturation" -> saturation,
    "pathFlows" -> pathFlows, "pathCosts" -> pathCosts,
    "allPaths" -> allPaths, "odGroups" -> odGroups,
    "odPairs" -> AberdeenOD, "objective" -> sol[[1]]
  |>
];

PlotAberdeenNetwork[res_Association] := Module[
  {edgeList, linkFlows, sat, maxSat, maxFlow, edgeStyles, vertexPos,
   vertexLabels, networkPlot, topSorted, barPlot, roadNames, flows,
   topIdx, topRoads, topSat, top, nEdges},

  edgeList = DirectedEdge[#[[1]], #[[2]]] & /@ AberdeenEdges;
  linkFlows = res["linkFlows"];
  sat = res["saturation"];
  maxFlow = Max[linkFlows];
  maxSat = Max[1.5, Max[sat]*1.05];

  edgeStyles = MapThread[
    #1 -> Directive[
      Thickness[0.003 + 0.015 #2/maxFlow],
      ColorData[{"TemperatureMap", {0, 1.5}}][#3]
    ] &,
    {edgeList, linkFlows, sat}
  ];

  vertexPos = Thread[
    Keys[AberdeenNodes] -> Values[AberdeenNodes][[All, 2]]
  ];
  vertexLabels = KeyValueMap[
    #1 -> Placed[
      Column[{Style[ToString[#1], Bold, 10],
              Style[First[#2], 6, Gray]},
             Spacings -> 0.1],
      Above
    ] &,
    AberdeenNodes
  ];

  networkPlot = Graph[
    Keys[AberdeenNodes], edgeList,
    VertexCoordinates -> vertexPos,
    VertexSize -> 0.07,
    VertexStyle -> LightBlue,
    VertexLabels -> vertexLabels,
    EdgeStyle -> edgeStyles,
    EdgeShapeFunction -> GraphElementData["CurvedArc"],
    ImageSize -> 700,
    PlotLabel -> "Aberdeen: link saturation (flow / capacity)"
  ];

  (* Top 10 congested links bar chart *)
  roadNames = res["roadNames"];
  flows = linkFlows;
  topIdx = Take[Reverse @ Ordering[sat], UpTo[10]];
  topRoads = StringTake[#, UpTo[22]] & /@ roadNames[[topIdx]];
  topSat = sat[[topIdx]];

  barPlot = BarChart[
    Reverse @ topSat,
    BarOrigin -> Left,
    ChartLabels -> Placed[Reverse @ topRoads, Before],
    ChartStyle -> (ColorData[{"TemperatureMap", {0, 1.5}}][#] & /@ Reverse[topSat]),
    Frame -> True, FrameLabel -> {"Saturation (flow/capacity)", None},
    PlotLabel -> "Top 10 most congested links",
    Epilog -> {Red, Dashed,
      Line[{{1.0, -0.5}, {1.0, Length[topIdx] + 0.5}}],
      Text[Style["capacity", Red, 8], {1.02, Length[topIdx] + 0.3}, {-1, 0}]
    },
    ImageSize -> 500, AspectRatio -> 1
  ];

  GraphicsRow[{networkPlot, barPlot}, ImageSize -> 1300, Spacings -> 30]
];

PlotAberdeenNetwork[res_Association, path_String] := Module[{g = PlotAberdeenNetwork[res]},
  Export[path, g, ImageResolution -> 150]; path];

End[];
EndPackage[];
