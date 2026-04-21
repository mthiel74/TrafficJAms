(* ::Package:: *)

(* Queueing.wl -- M/D/1 queueing model for signalised corridors.
   Port of trafficjams/queueing.py. *)

BeginPackage["Queueing`"];

MD1QueueLength::usage = "MD1QueueLength[rho] = rho^2 / (2(1-rho)).";
MD1WaitingTime::usage = "MD1WaitingTime[rho, mu] = rho / (2 mu (1-rho)).";
SimulateQueueing::usage = "SimulateQueueing[opts] returns MD1 queue length and waiting time as a function of utilisation.";
PlotQueueing::usage = "PlotQueueing[res] renders the four-panel summary plot.";

Begin["`Private`"];

MD1QueueLength[rho_] := rho^2/(2 (1 - rho));
MD1WaitingTime[rho_, mu_] := rho/(2 mu (1 - rho));

Options[SimulateQueueing] = {"mu" -> 0.5, "nIntersections" -> 4,
  "lambdaMin" -> 0.05, "lambdaMax" -> 0.48, "nSamples" -> 100};

SimulateQueueing[OptionsPattern[]] := Module[
  {mu, n, lambdaMin, lambdaMax, nSamples, lambda, rho, Lq, Wq},
  mu = OptionValue["mu"]; n = OptionValue["nIntersections"];
  lambdaMin = OptionValue["lambdaMin"]; lambdaMax = OptionValue["lambdaMax"];
  nSamples = OptionValue["nSamples"];

  lambda = N @ Subdivide[lambdaMin, lambdaMax, nSamples - 1];
  rho = lambda/mu;
  Lq = MD1QueueLength /@ rho;
  Wq = MD1WaitingTime[#, mu] & /@ rho;

  <|
    "lambda" -> lambda, "mu" -> mu, "rho" -> rho,
    "queueLength" -> Lq, "waitingTime" -> Wq,
    "corridorQueue" -> n Lq, "corridorDelay" -> n Wq,
    "nIntersections" -> n
  |>
];

PlotQueueing[res_Association] := Module[
  {rho, Lq, Wq, cq, cd, n, commonOpts, p1, p2, p3, p4, yMax},

  rho = res["rho"]; Lq = res["queueLength"]; Wq = res["waitingTime"];
  cq = res["corridorQueue"]; cd = res["corridorDelay"];
  n = res["nIntersections"];

  commonOpts = Sequence[Frame -> True, GridLines -> Automatic,
    GridLinesStyle -> Directive[LightGray, Dashed], ImageSize -> 500,
    AspectRatio -> 0.75];

  p1 = ListLinePlot[Transpose[{rho, Lq}],
    PlotStyle -> Blue, FrameLabel -> {"Utilisation \[Rho] = \[Lambda]/\[Mu]", "Expected queue length Lq"},
    PlotLabel -> "M/D/1: Queue length vs utilisation",
    PlotRange -> {{0, 1}, {0, Min[50, Max[Lq] 1.1]}},
    Epilog -> {Red, Dashed, Line[{{0.85, 0}, {0.85, 50}}]},
    commonOpts];

  p2 = ListLinePlot[Transpose[{rho, Wq}],
    PlotStyle -> Red, FrameLabel -> {"Utilisation \[Rho]", "Expected waiting time (s)"},
    PlotLabel -> "M/D/1: Waiting time vs utilisation",
    PlotRange -> {{0, 1}, {0, Min[100, Max[Wq] 1.1]}},
    commonOpts];

  p3 = ListLinePlot[Transpose[{rho, cq}],
    PlotStyle -> Darker[Green],
    FrameLabel -> {"Utilisation \[Rho]", "Total queue length"},
    PlotLabel -> Row[{"Corridor (", n, " intersections): total queue"}],
    PlotRange -> {{0, 1}, {0, Min[200, Max[cq] 1.1]}},
    commonOpts];

  p4 = ListLinePlot[Transpose[{rho, cd}],
    PlotStyle -> Purple,
    FrameLabel -> {"Utilisation \[Rho]", "Total corridor delay (s)"},
    PlotLabel -> "Corridor: total delay vs utilisation",
    PlotRange -> {{0, 1}, {0, Min[400, Max[cd] 1.1]}},
    commonOpts];

  GraphicsGrid[{{p1, p2}, {p3, p4}}, ImageSize -> 1100, Spacings -> 20]
];

PlotQueueing[res_Association, path_String] := Module[{g = PlotQueueing[res]},
  Export[path, g, ImageResolution -> 150]; path];

End[];
EndPackage[];
