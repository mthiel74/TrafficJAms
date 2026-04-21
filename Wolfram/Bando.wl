(* ::Package:: *)

(* Bando.wl -- Optimal Velocity Model (Bando OVM) on a circular road.
   Port of trafficjams/bando.py.

   V_opt(s) = v_max * (tanh(s/s_c - 2) + tanh(2)) / (1 + tanh(2))
   v'_a    = kappa * (V_opt(s_a) - v_a)
*)

BeginPackage["Bando`"];

OptimalVelocity::usage = "OptimalVelocity[s, vmax, sc] returns the Bando optimal-velocity target at headway s.";
SimulateBando::usage = "SimulateBando[opts] runs the Bando OVM on a ring road.";
PlotBando::usage = "PlotBando[result] draws trajectory/mean-speed plots.";
AnimateBando::usage = "AnimateBando[result, path] exports a ring-road animation to an animated GIF at path.";

Begin["`Private`"];

OptimalVelocity[s_, vmax_, sc_] := vmax*(Tanh[s/sc - 2] + Tanh[2])/(1 + Tanh[2]);

bandoStep[pos_, vel_, roadLength_, params_] := Module[
  {leaderPos, gaps, vOpt, accel, newVel, newPos, kappa, vmax, sc, dt},
  {kappa, vmax, sc, dt} = Lookup[params, {"kappa", "vmax", "sc", "dt"}];
  leaderPos = RotateLeft[pos];
  gaps = Mod[leaderPos - pos, roadLength];
  vOpt = OptimalVelocity[gaps, vmax, sc];
  accel = kappa (vOpt - vel);
  newVel = Clip[vel + accel dt, {0, Infinity}];
  newPos = Mod[pos + newVel dt, roadLength];
  {newPos, newVel}
];

Options[SimulateBando] = {
  "nVehicles" -> 40,
  "roadLength" -> 800.0,
  "T" -> 200.0,
  "dt" -> 0.1,
  "kappa" -> 1.0,
  "vmax" -> 15.0,
  "sc" -> 25.0,
  "perturbation" -> 5.0     (* backward displacement of vehicle 0 (m) *)
};

SimulateBando[OptionsPattern[]] := Module[
  {n, L, T, dt, kappa, vmax, sc, perturb, nt, spacing, pos, vel, posHist,
   velHist, times, params, sampleEvery, step, newState},
  n = OptionValue["nVehicles"];
  L = OptionValue["roadLength"];
  T = OptionValue["T"];
  dt = OptionValue["dt"];
  kappa = OptionValue["kappa"];
  vmax = OptionValue["vmax"];
  sc = OptionValue["sc"];
  perturb = OptionValue["perturbation"];

  nt = Floor[T/dt];
  spacing = L/n;
  pos = N @ Table[i spacing, {i, 0, n - 1}];
  vel = ConstantArray[N @ OptimalVelocity[spacing, vmax, sc], n];

  pos[[1]] = Mod[pos[[1]] - perturb, L]; (* backward displacement of vehicle 0 *)

  params = <|"kappa" -> kappa, "vmax" -> vmax, "sc" -> sc, "dt" -> dt|>;

  posHist = {pos}; velHist = {vel}; times = {0.0};
  sampleEvery = Max[1, Quotient[nt, 200]];

  Do[
    newState = bandoStep[pos, vel, L, params];
    pos = newState[[1]]; vel = newState[[2]];
    If[Mod[step, sampleEvery] == 0,
      AppendTo[posHist, pos]; AppendTo[velHist, vel]; AppendTo[times, step dt];
    ],
    {step, 1, nt}
  ];

  <|
    "times" -> times,
    "positions" -> posHist,
    "velocities" -> velHist,
    "roadLength" -> L,
    "nVehicles" -> n,
    "kappa" -> kappa
  |>
];

PlotBando[res_Association] := Module[
  {times, pos, vel, n, ptsByVeh, trajPlot, meanV, stdV, statsPlot},

  times = res["times"]; pos = res["positions"]; vel = res["velocities"];
  n = res["nVehicles"];

  ptsByVeh = Table[
    Transpose[{times, pos[[All, i]], vel[[All, i]]}],
    {i, 1, n}
  ];

  trajPlot = ListPlot[
    Table[
      Style[{#[[1]], #[[2]]} & /@ ptsByVeh[[i]], PointSize[0.003],
        ColorData["TemperatureMap"][1 - Mean[ptsByVeh[[i, All, 3]]]/15.]],
      {i, 1, n}
    ],
    PlotRange -> {{0, Max[times]}, {0, res["roadLength"]}},
    Frame -> True, FrameLabel -> {"Time (s)", "Position (m)"},
    PlotLabel -> Row[{"Bando OVM: Phantom Jams (\[Kappa]=", res["kappa"], ")"}],
    ImageSize -> 520, AspectRatio -> 0.8
  ];

  meanV = Mean /@ vel;
  stdV = StandardDeviation /@ vel;

  statsPlot = ListLinePlot[
    {
      Transpose[{times, meanV}],
      Transpose[{times, meanV + stdV}],
      Transpose[{times, meanV - stdV}]
    },
    Frame -> True, FrameLabel -> {"Time (s)", "Speed (m/s)"},
    PlotLabel -> "Speed Statistics Over Time",
    PlotLegends -> {"Mean", "+1 std", "-1 std"},
    ImageSize -> 520, AspectRatio -> 0.8
  ];

  GraphicsRow[{trajPlot, statsPlot}, ImageSize -> 1100]
];

PlotBando[res_Association, path_String] := Module[{g = PlotBando[res]},
  Export[path, g, ImageResolution -> 150]; path];

speedColour[v_] := Blend[
  {RGBColor[0.8, 0.1, 0.1], RGBColor[1.0, 0.85, 0.2], RGBColor[0.15, 0.7, 0.2]},
  Clip[v/15.0, {0, 1}]
];

bandoFrame[res_Association, k_Integer] := Module[
  {n, L, pos, vel, angles, pts, t, meanV},
  n = res["nVehicles"]; L = res["roadLength"];
  pos = res["positions"][[k]]; vel = res["velocities"][[k]];
  t = res["times"][[k]]; meanV = Mean[vel];
  angles = 2 Pi pos/L;
  pts = MapThread[
    {speedColour[#2], Disk[{Cos[#1], Sin[#1]}, 0.06]} &,
    {angles, vel}
  ];
  Graphics[
    {
      {GrayLevel[0.85], Thickness[0.02], Circle[{0, 0}, 1]},
      {Black, EdgeForm[Darker[Gray]], pts},
      Text[Style[Row[{"\[LeftAngleBracket]v\[RightAngleBracket] = ",
          NumberForm[meanV, {4, 2}], " m/s"}], 11], {0, 0}]
    },
    PlotRange -> 1.2 {{-1, 1}, {-1, 1}},
    Background -> GrayLevel[0.97], ImageSize -> 360,
    PlotLabel -> Row[{"Bando OVM  \[Kappa] = ", res["kappa"],
      "  t = ", NumberForm[N[t], {4, 1}], " s"}]
  ]
];

Options[AnimateBando] = {"frameStep" -> 2, "displayDuration" -> 0.07};
AnimateBando[res_Association, path_String, OptionsPattern[]] := Module[
  {frames, step = OptionValue["frameStep"], nFrames = Length[res["times"]]},
  frames = Table[bandoFrame[res, k], {k, 1, nFrames, step}];
  Export[path, frames, "GIF",
    "AnimationRepetitions" -> Infinity,
    "DisplayDurations" -> OptionValue["displayDuration"]];
  path
];

End[];
EndPackage[];
