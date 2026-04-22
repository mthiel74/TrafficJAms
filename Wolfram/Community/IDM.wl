(* ::Package:: *)

(* IDM.wl -- Intelligent Driver Model on a circular road.
   Port of trafficjams/idm.py.

   Usage:
     Get["IDM.wl"];
     res = SimulateIDM[];
     PlotIDM[res, "results/idm.png"];
*)

BeginPackage["IDM`"];

SimulateIDM::usage = "SimulateIDM[opts] runs the IDM on a ring road and returns an Association with keys times, positions, velocities, roadLength, nVehicles.";
PlotIDM::usage = "PlotIDM[result] returns trajectory and speed-profile plots. PlotIDM[result, path] also exports to path.";

Begin["`Private`"];

(* Single IDM step: pos, vel are length-n vectors.
   Returns {newPos, newVel}. *)
idmStep[pos_, vel_, roadLength_, params_] := Module[
  {n = Length[pos], leaderPos, leaderVel, gaps, dv, sStar, accel, newVel, newPos,
   v0, s0, Tgap, a, b, delta, vehLen, dt},
  {v0, s0, Tgap, a, b, delta, vehLen, dt} =
    Lookup[params, {"v0", "s0", "T", "a", "b", "delta", "vehLen", "dt"}];

  leaderPos = RotateLeft[pos];
  leaderVel = RotateLeft[vel];

  gaps = Mod[leaderPos - pos, roadLength] - vehLen;
  gaps = Clip[gaps, {0.1, Infinity}];

  dv = vel - leaderVel;

  sStar = s0 + vel*Tgap + vel*dv/(2 Sqrt[a*b]);
  sStar = Clip[sStar, {s0, Infinity}];

  accel = a*(1 - (vel/v0)^delta - (sStar/gaps)^2);

  newVel = Clip[vel + accel*dt, {0, Infinity}];
  newPos = Mod[pos + newVel*dt, roadLength];

  {newPos, newVel}
];

Options[SimulateIDM] = {
  "nVehicles" -> 50,
  "roadLength" -> 1000.0,
  "T" -> 120.0,
  "dt" -> 0.1,
  "v0" -> 15.0,
  "s0" -> 2.0,
  "THeadway" -> 1.5,
  "a" -> 1.0,
  "b" -> 1.5,
  "delta" -> 4,
  "vehLen" -> 5.0
};

SimulateIDM[OptionsPattern[]] := Module[
  {n, L, T, dt, v0, s0, Tg, a, b, delta, vehLen, nt, spacing,
   pos, vel, posHist, velHist, times, params, sampleEvery, step, newState},

  n = OptionValue["nVehicles"];
  L = OptionValue["roadLength"];
  T = OptionValue["T"];
  dt = OptionValue["dt"];
  v0 = OptionValue["v0"];
  s0 = OptionValue["s0"];
  Tg = OptionValue["THeadway"];
  a = OptionValue["a"];
  b = OptionValue["b"];
  delta = OptionValue["delta"];
  vehLen = OptionValue["vehLen"];

  nt = Floor[T/dt];
  spacing = L/n;
  pos = N @ Table[i spacing, {i, 0, n - 1}];
  vel = ConstantArray[N[v0], n];
  vel[[1]] = 0.5 vel[[1]];  (* perturbation *)

  params = <|
    "v0" -> v0, "s0" -> s0, "T" -> Tg, "a" -> a, "b" -> b,
    "delta" -> delta, "vehLen" -> vehLen, "dt" -> dt
  |>;

  posHist = {pos};
  velHist = {vel};
  times = {0.0};
  sampleEvery = Max[1, Quotient[nt, 200]];

  Do[
    newState = idmStep[pos, vel, L, params];
    pos = newState[[1]]; vel = newState[[2]];
    If[Mod[step, sampleEvery] == 0,
      AppendTo[posHist, pos];
      AppendTo[velHist, vel];
      AppendTo[times, step dt];
    ],
    {step, 1, nt}
  ];

  <|
    "times" -> times,
    "positions" -> posHist,
    "velocities" -> velHist,
    "roadLength" -> L,
    "nVehicles" -> n
  |>
];

PlotIDM[res_Association] := Module[
  {times, pos, vel, n, ptsByVeh, trajPlot, speedPlot, sampleVeh},

  times = res["times"];
  pos = res["positions"];
  vel = res["velocities"];
  n = Min[res["nVehicles"], 50];

  (* Build coloured space-time scatter: for each vehicle, pair (t, x_i(t)) coloured by v_i(t). *)
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
    PlotLabel -> "IDM: Vehicle Trajectories (circular road)",
    ImageSize -> 520, AspectRatio -> 0.8
  ];

  sampleVeh = Select[{1, 11, 26, 41}, # <= res["nVehicles"] &];
  speedPlot = ListLinePlot[
    Table[Transpose[{times, vel[[All, i]]}], {i, sampleVeh}],
    Frame -> True, FrameLabel -> {"Time (s)", "Speed (m/s)"},
    PlotLabel -> "IDM: Speed Profiles",
    PlotLegends -> (("Vehicle " <> ToString[# - 1]) & /@ sampleVeh),
    ImageSize -> 520, AspectRatio -> 0.8
  ];

  GraphicsRow[{trajPlot, speedPlot}, ImageSize -> 1100]
];

PlotIDM[res_Association, path_String] := Module[{g = PlotIDM[res]},
  Export[path, g, ImageResolution -> 150];
  path
];

End[];
EndPackage[];
