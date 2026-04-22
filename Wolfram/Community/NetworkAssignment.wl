(* ::Package:: *)

(* NetworkAssignment.wl -- Wardrop-equilibrium static traffic assignment via the
   Beckmann formulation with BPR link-cost functions. Port of
   trafficjams/network_assignment.py. *)

BeginPackage["NetworkAssignment`"];

BPRCost::usage = "BPRCost[flow, t0, capacity, alpha, beta] Bureau of Public Roads link-cost function.";
BPRIntegral::usage = "BPRIntegral[flow, t0, capacity, alpha, beta] integral used in the Beckmann objective.";
SimulateAssignment::usage = "SimulateAssignment[] solves a Wardrop equilibrium on an Aberdeen-inspired 8-node network.";
PlotAssignment::usage = "PlotAssignment[res] returns a network flow map + bar chart.";

Begin["`Private`"];

BPRCost[flow_, t0_, capacity_, alpha_: 0.15, beta_: 4.0] :=
  t0 (1 + alpha (flow/capacity)^beta);

BPRIntegral[flow_, t0_, capacity_, alpha_: 0.15, beta_: 4.0] :=
  t0 (flow + alpha capacity/(beta + 1) (flow/capacity)^(beta + 1));

SimulateAssignment[] := Module[
  {nodes, nodeLabels, edges, fft, cap, totalDemand, paths, nEdges, nPaths,
   Delta, vars, beckmann, constraints, sol, pathFlows, linkFlows, linkCosts,
   pathCosts},

  nodes = Range[0, 7];
  nodeLabels = <|
    0 -> "Bridge of Don", 1 -> "King St / A956", 2 -> "Mounthooly",
    3 -> "Union St / Market St", 4 -> "Bridge of Dee",
    5 -> "Anderson Drive North", 6 -> "Anderson Drive South",
    7 -> "A90 North"
  |>;

  (* edges: {from, to, freeFlowTime, capacity} *)
  edges = {
    {7, 0, 5, 2000}, {0, 1, 4, 1500}, {1, 2, 3, 1200}, {2, 3, 2, 1000},
    {3, 4, 5, 1200}, {7, 5, 6, 1800}, {5, 6, 4, 1600}, {6, 4, 3, 1400},
    {5, 2, 3, 800},  {0, 5, 5, 900}
  };
  nEdges = Length[edges];
  fft = N @ edges[[All, 3]];
  cap = N @ edges[[All, 4]];
  totalDemand = 3000.0;

  (* path edge-index lists (1-based) *)
  paths = {
    {1, 2, 3, 4, 5},
    {6, 7, 8},
    {1, 10, 7, 8},
    {6, 9, 4, 5}
  };
  nPaths = Length[paths];

  Delta = ConstantArray[0, {nEdges, nPaths}];
  Do[Do[Delta[[e, p]] = 1, {e, paths[[p]]}], {p, nPaths}];

  vars = Array[f, nPaths];
  linkFlows = Delta . vars;
  beckmann = Sum[
    BPRIntegral[linkFlows[[e]], fft[[e]], cap[[e]]],
    {e, nEdges}
  ];

  constraints = Join[
    {Total[vars] == totalDemand},
    Thread[vars >= 0]
  ];

  sol = FindMinimum[
    {beckmann, constraints}, vars,
    Method -> "InteriorPoint"
  ];

  pathFlows = vars /. sol[[2]];
  linkFlows = Delta . pathFlows;
  linkCosts = MapThread[BPRCost[#1, #2, #3] &, {linkFlows, fft, cap}];
  pathCosts = Transpose[Delta] . linkCosts;

  <|
    "nodes" -> nodes,
    "nodeLabels" -> nodeLabels,
    "edges" -> edges,
    "linkFlows" -> linkFlows,
    "linkCosts" -> linkCosts,
    "pathFlows" -> pathFlows,
    "pathCosts" -> pathCosts,
    "paths" -> paths,
    "totalDemand" -> totalDemand,
    "objective" -> sol[[1]]
  |>
];

PlotAssignment[res_Association] := Module[
  {edges, linkFlows, pathFlows, pathCosts, maxFlow, edgeList, weights, edgeLabels,
   g, pos, nodeLabels, bar},

  edges = res["edges"]; linkFlows = res["linkFlows"];
  pathFlows = res["pathFlows"]; pathCosts = res["pathCosts"];
  nodeLabels = res["nodeLabels"];
  maxFlow = Max[linkFlows];

  edgeList = DirectedEdge[#[[1]], #[[2]]] & /@ edges;
  weights = 1 + 4 linkFlows/maxFlow;
  edgeLabels = MapThread[#1 -> NumberForm[#2, 4] &, {edgeList, linkFlows}];

  pos = <|
    7 -> {0, 2}, 0 -> {2, 3}, 1 -> {3, 2.5}, 2 -> {3, 1.5},
    3 -> {3, 0.5}, 4 -> {2, -0.5}, 5 -> {1, 2}, 6 -> {1, 0.5}
  |>;

  g = Graph[Keys[nodeLabels], edgeList,
    VertexCoordinates -> (pos[#] & /@ Keys[nodeLabels]),
    VertexLabels -> KeyValueMap[#1 -> Placed[Style[#2, 8, Black], Above] &, nodeLabels],
    VertexSize -> 0.18,
    VertexStyle -> LightBlue,
    EdgeStyle -> MapThread[
      #1 -> Directive[Thickness[0.004 #2], ColorData["Rainbow"][#3/maxFlow]] &,
      {edgeList, weights, linkFlows}
    ],
    EdgeLabels -> edgeLabels,
    EdgeLabelStyle -> Directive[7, Darker[Gray]],
    ImageSize -> 560,
    PlotLabel -> "Aberdeen network: equilibrium flows"
  ];

  bar = BarChart[pathFlows,
    ChartLabels -> Placed[Table["Path " <> ToString[i], {i, Length[pathFlows]}], Below],
    ChartStyle -> RGBColor[0.27, 0.51, 0.71],
    PlotLabel -> "Path flow distribution (Wardrop equilibrium)",
    Frame -> True, FrameLabel -> {None, "Flow (vehicles)"},
    ImageSize -> 520,
    Epilog -> (
      Table[
        Text[
          Style[Row[{"cost=", NumberForm[pathCosts[[i]], {5, 1}]}], 9],
          {i, pathFlows[[i]] + 60}, {0, 0}],
        {i, Length[pathFlows]}
      ]
    )
  ];

  GraphicsRow[{g, bar}, ImageSize -> 1150]
];

PlotAssignment[res_Association, path_String] := Module[{g = PlotAssignment[res]},
  Export[path, g, ImageResolution -> 150]; path];

End[];
EndPackage[];
