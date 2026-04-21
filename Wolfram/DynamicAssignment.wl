(* ::Package:: *)

(* DynamicAssignment.wl -- period-by-period Wardrop assignment under a
   time-varying demand profile with carry-over congestion between
   periods. Port of trafficjams/dynamic_assignment.py. *)

BeginPackage["DynamicAssignment`"];

SimulateDynamicAssignment::usage = "SimulateDynamicAssignment[opts] runs a sequential equilibrium assignment across n_periods under a rush-hour demand profile.";
PlotDynamicAssignment::usage = "PlotDynamicAssignment[res] renders demand, path flows, key link flows, and link-cost heatmap.";

Begin["`Private`"];

bprCost[f_, fft_, cap_, alpha_: 0.15, beta_: 4.0] :=
  fft (1 + alpha (f/cap)^beta);
bprIntegral[f_, fft_, cap_, alpha_: 0.15, beta_: 4.0] :=
  fft (f + alpha cap/(beta + 1) (f/cap)^(beta + 1));

Options[SimulateDynamicAssignment] = {
  "nPeriods" -> 12,
  "periodLength" -> 5,
  "baseDemand" -> 3000.0
};

SimulateDynamicAssignment[OptionsPattern[]] := Module[
  {nPeriods, periodLen, base, edges, fft, cap, paths, nEdges, nPaths,
   delta, t, demandProfile, allLinkFlows, allPathFlows, allLinkCosts,
   residual, period, demand, effCap, vars, obj, constraints, sol, pf,
   lf, lc},

  nPeriods = OptionValue["nPeriods"];
  periodLen = OptionValue["periodLength"];
  base = OptionValue["baseDemand"];

  (* Same simplified Aberdeen sub-network as the static model *)
  edges = {{7,0,5,2000},{0,1,4,1500},{1,2,3,1200},{2,3,2,1000},{3,4,5,1200},
           {7,5,6,1800},{5,6,4,1600},{6,4,3,1400},{5,2,3,800},{0,5,5,900}};
  nEdges = Length[edges];
  fft = N @ edges[[All, 3]];
  cap = N @ edges[[All, 4]];

  paths = {{1,2,3,4,5}, {6,7,8}, {1,10,7,8}, {6,9,4,5}};
  nPaths = Length[paths];
  delta = ConstantArray[0, {nEdges, nPaths}];
  Do[Do[delta[[e, p]] = 1, {e, paths[[p]]}], {p, nPaths}];

  (* Gaussian rush-hour profile peaking around 40% through the simulation *)
  t = Range[0, nPeriods - 1];
  demandProfile = base (0.3 + 0.7 Exp[-0.5 ((t - nPeriods 0.4)/(nPeriods 0.2))^2]);

  allLinkFlows = ConstantArray[0., {nPeriods, nEdges}];
  allPathFlows = ConstantArray[0., {nPeriods, nPaths}];
  allLinkCosts = ConstantArray[0., {nPeriods, nEdges}];
  residual = ConstantArray[0., nEdges];

  Do[
    demand = demandProfile[[period + 1]];
    (* capacity reduced by lingering congestion *)
    effCap = MapThread[Max[#1 - #2 0.3, #1 0.5] &, {cap, residual}];

    vars = Array[f, nPaths];
    obj = Total @ Table[
      bprIntegral[(delta . vars)[[e]] + residual[[e]] 0.2,
                  fft[[e]], effCap[[e]]],
      {e, nEdges}
    ];
    constraints = Join[
      {Total[vars] == demand},
      Thread[vars >= 0],
      Thread[vars <= demand]
    ];
    sol = FindMinimum[
      {obj, constraints},
      Thread[{vars, ConstantArray[demand/nPaths, nPaths]}],
      Method -> "InteriorPoint", MaxIterations -> 500
    ];
    pf = Clip[vars /. sol[[2]], {0, Infinity}];
    lf = delta . pf;
    lc = MapThread[bprCost[#1, #2, #3] &, {lf, fft, cap}];

    allPathFlows[[period + 1]] = pf;
    allLinkFlows[[period + 1]] = lf;
    allLinkCosts[[period + 1]] = lc;
    residual = lf 0.5,
    {period, 0, nPeriods - 1}
  ];

  <|
    "nPeriods" -> nPeriods,
    "periodLength" -> periodLen,
    "demandProfile" -> demandProfile,
    "linkFlows" -> allLinkFlows,
    "linkCosts" -> allLinkCosts,
    "pathFlows" -> allPathFlows,
    "edges" -> edges,
    "fft" -> fft,
    "cap" -> cap
  |>
];

PlotDynamicAssignment[res_Association] := Module[
  {periods, demand, pf, lf, lc, nPaths, keyLinks, edgeNames, demandPlot,
   pathPlot, linkPlot, heatPlot, pLen},

  pLen = res["periodLength"];
  periods = pLen Range[0, res["nPeriods"] - 1];
  demand = res["demandProfile"];
  pf = res["pathFlows"];
  lf = res["linkFlows"];
  lc = res["linkCosts"];
  nPaths = Dimensions[pf][[2]];

  demandPlot = ListLinePlot[
    Transpose[{periods, demand}],
    PlotStyle -> {Blue, Thick}, PlotMarkers -> Automatic,
    Filling -> Bottom, FillingStyle -> Directive[Blue, Opacity[0.2]],
    Frame -> True, FrameLabel -> {"Time (min)", "Total demand (veh/period)"},
    PlotLabel -> "Morning rush hour demand profile",
    GridLines -> Automatic, GridLinesStyle -> Directive[LightGray, Dashed],
    ImageSize -> 460, AspectRatio -> 0.7
  ];

  pathPlot = ListLinePlot[
    Table[Transpose[{periods, pf[[All, i]]}], {i, nPaths}],
    PlotMarkers -> Automatic,
    Frame -> True, FrameLabel -> {"Time (min)", "Path flow"},
    PlotLabel -> "Path flow evolution",
    PlotLegends -> (("Path " <> ToString[#]) & /@ Range[nPaths]),
    GridLines -> Automatic, GridLinesStyle -> Directive[LightGray, Dashed],
    ImageSize -> 460, AspectRatio -> 0.7
  ];

  keyLinks = {1, 2, 6, 7};
  edgeNames = {"A90\[RightArrow]Bridge of Don", "Bridge of Don\[RightArrow]King St",
               "A90\[RightArrow]Anderson Dr N", "Anderson Dr N\[RightArrow]S"};
  linkPlot = ListLinePlot[
    Table[Transpose[{periods, lf[[All, keyLinks[[i]]]]}], {i, Length[keyLinks]}],
    PlotMarkers -> Automatic,
    Frame -> True, FrameLabel -> {"Time (min)", "Link flow"},
    PlotLabel -> "Key link flows over time",
    PlotLegends -> edgeNames,
    GridLines -> Automatic, GridLinesStyle -> Directive[LightGray, Dashed],
    ImageSize -> 460, AspectRatio -> 0.7
  ];

  heatPlot = ArrayPlot[
    Transpose @ lc,
    DataReversed -> True,
    ColorFunction -> "SolarColors",
    ColorFunctionScaling -> True,
    Frame -> True,
    FrameLabel -> {"Time period", "Link index"},
    PlotLabel -> "Link travel costs over time",
    PlotLegends -> Automatic,
    ImageSize -> 460, AspectRatio -> 0.7
  ];

  Column[{
    Style["Dynamic traffic assignment: morning rush hour", 14, Bold],
    GraphicsGrid[{{demandPlot, pathPlot}, {linkPlot, heatPlot}},
                 ImageSize -> 1100, Spacings -> 20]
  }]
];

PlotDynamicAssignment[res_Association, path_String] := Module[{g = PlotDynamicAssignment[res]},
  Export[path, g, ImageResolution -> 150]; path];

End[];
EndPackage[];
