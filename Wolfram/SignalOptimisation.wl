(* ::Package:: *)

(* SignalOptimisation.wl -- optimise green phase splits for a signalised
   corridor. Port of trafficjams/signal_optimisation.py.

   Objective per intersection: lambda (Wq + d_uniform) where
     mu      = saturation_flow * g / cycle
     Wq      = rho / (2 mu (1 - rho))   (M/D/1 waiting)
     d_unif  = (cycle - g)^2 / (2 cycle) (Webster uniform delay)
*)

BeginPackage["SignalOptimisation`"];

CorridorDelay::usage = "CorridorDelay[greens, arrivals, cycle, satFlow] returns total delay over a signalised corridor.";
SimulateSignalOpt::usage = "SimulateSignalOpt[opts] returns uniform vs optimised green splits and their delays.";
PlotSignalOpt::usage = "PlotSignalOpt[res] draws the green-split / delay / sensitivity comparison.";

Begin["`Private`"];

CorridorDelay[greens_List, arrivals_List, cycle_, satFlow_] := Module[
  {total = 0., n = Length[greens], g, lam, mu, rho, Wq, dUnif, red, i},
  Do[
    g = greens[[i]]; lam = arrivals[[i]];
    mu = satFlow g/cycle;
    If[mu <= 0 || mu <= lam, total += 1.*^6; Continue[]];
    rho = lam/mu;
    If[rho >= 1, total += 1.*^6; Continue[]];
    Wq = rho/(2 mu (1 - rho));
    red = cycle - g;
    dUnif = red^2/(2 cycle);
    total += lam (Wq + dUnif),
    {i, n}
  ];
  total
];

Options[SimulateSignalOpt] = {
  "nIntersections" -> 5, "cycleTime" -> 90,
  "arrivals" -> {0.08, 0.15, 0.22, 0.17, 0.06},
  "satFlow" -> 0.5,
  "uniformGreen" -> 50, "totalBudget" -> Automatic,
  "minGreen" -> 20, "maxGreenMargin" -> 15
};

SimulateSignalOpt[OptionsPattern[]] := Module[
  {n, cycle, arr, satFlow, uGreen, budget, minG, maxG, vars, uniformDelay,
   objective, constraints, sol, optimalGreen, optimalDelay, sweep, gAxis, i,
   delayAt, busiest, improvement},

  n = OptionValue["nIntersections"];
  cycle = N @ OptionValue["cycleTime"];
  arr = Take[N @ OptionValue["arrivals"], n];
  satFlow = N @ OptionValue["satFlow"];
  minG = N @ OptionValue["minGreen"];
  maxG = cycle - N @ OptionValue["maxGreenMargin"];
  uGreen = ConstantArray[N @ OptionValue["uniformGreen"], n];
  budget = Replace[OptionValue["totalBudget"], Automatic -> n 45.];

  uniformDelay = CorridorDelay[uGreen, arr, cycle, satFlow];

  vars = Array[g, n];
  objective = CorridorDelay[vars, arr, cycle, satFlow];
  constraints = Join[
    {Total[vars] == budget},
    Thread[vars >= minG],
    Thread[vars <= maxG]
  ];
  sol = FindMinimum[
    {objective, constraints}, Thread[{vars, uGreen}],
    Method -> "InteriorPoint", MaxIterations -> 500
  ];
  optimalGreen = vars /. sol[[2]];
  optimalDelay = sol[[1]];

  (* sensitivity sweep for each intersection *)
  gAxis = N @ Subdivide[minG, maxG, 49];
  sweep = Table[
    delayAt = Table[
      With[{splits = ReplacePart[optimalGreen, i -> g0]},
        CorridorDelay[splits, arr, cycle, satFlow]
      ],
      {g0, gAxis}
    ];
    {gAxis, delayAt},
    {i, n}
  ];

  busiest = First @ Ordering[-arr];
  improvement = (1 - optimalDelay/uniformDelay) 100;

  <|
    "nIntersections" -> n,
    "cycleTime" -> cycle,
    "arrivals" -> arr,
    "uniformGreen" -> uGreen,
    "uniformDelay" -> uniformDelay,
    "optimalGreen" -> optimalGreen,
    "optimalDelay" -> optimalDelay,
    "sweep" -> sweep,
    "busiest" -> busiest,
    "improvementPct" -> improvement
  |>
];

PlotSignalOpt[res_Association] := Module[
  {n, arr, uG, oG, lbl, greenBar, delayBar, sensPlot, busiest, sw},

  n = res["nIntersections"];
  arr = res["arrivals"];
  uG = res["uniformGreen"];
  oG = res["optimalGreen"];
  busiest = res["busiest"];

  lbl = Table[
    Column[{"Int " <> ToString[i], "\[Lambda]=" <> ToString[NumberForm[arr[[i]], {3, 2}]]}, Alignment -> Center],
    {i, n}
  ];

  greenBar = BarChart[
    Transpose[{uG, oG}],
    ChartLabels -> {Placed[lbl, Below], None},
    ChartLegends -> {"Uniform", "Optimised"},
    ChartStyle -> {RGBColor[0.94, 0.50, 0.50], Darker[Green]},
    PlotLabel -> "Green phase allocation",
    Frame -> True, FrameLabel -> {None, "Green time (s)"},
    ImageSize -> 420, AspectRatio -> 0.8
  ];

  delayBar = BarChart[
    {res["uniformDelay"], res["optimalDelay"]},
    ChartLabels -> Placed[{"Uniform", "Optimised"}, Below],
    ChartStyle -> {RGBColor[0.94, 0.50, 0.50], Darker[Green]},
    PlotLabel -> Row[{"Total delay (", NumberForm[res["improvementPct"], {4, 1}], "% reduction)"}],
    Frame -> True, FrameLabel -> {None, "Total corridor delay (veh\[CenterDot]s)"},
    ImageSize -> 380, AspectRatio -> 0.8
  ];

  sw = res["sweep"][[busiest]];
  sensPlot = ListLinePlot[
    Transpose[sw],
    PlotStyle -> Blue, PlotRange -> {All, {0, Min[Max[sw[[2]]] 1.05, 50 Max[arr]]}},
    Frame -> True, FrameLabel -> {"Green time (s)", "Total corridor delay"},
    PlotLabel -> Row[{"Sensitivity: intersection ", busiest, " (busiest)"}],
    Epilog -> {
      {Green, Dashed, Line[{{oG[[busiest]], 0}, {oG[[busiest]], 10^6}}]},
      {Red, Dashed, Line[{{uG[[busiest]], 0}, {uG[[busiest]], 10^6}}]}
    },
    PlotLegends -> None,
    ImageSize -> 400, AspectRatio -> 0.8
  ];

  Column[{
    Style["Signal timing optimisation: Union St corridor", 14, Bold],
    GraphicsRow[{greenBar, delayBar, sensPlot}, ImageSize -> 1200, Spacings -> 15]
  }]
];

PlotSignalOpt[res_Association, path_String] := Module[{g = PlotSignalOpt[res]},
  Export[path, g, ImageResolution -> 150]; path];

End[];
EndPackage[];
