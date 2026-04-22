(* ::Package:: *)

(* NagelSchreckenberg.wl -- stochastic cellular automaton model.
   Port of trafficjams/nagel_schreckenberg.py.
   At each step: accelerate, brake to gap, randomly slow, move. *)

BeginPackage["NagelSchreckenberg`"];

SimulateNaSch::usage = "SimulateNaSch[opts] runs the NaSch CA on a periodic lattice.";
PlotNaSch::usage = "PlotNaSch[res] renders the space-time diagram.";

Begin["`Private`"];

Options[SimulateNaSch] = {
  "roadLength" -> 500, "nVehicles" -> 100,
  "vMax" -> 5, "pSlow" -> 0.3, "T" -> 300, "seed" -> 42
};

SimulateNaSch[OptionsPattern[]] := Module[
  {L, n, vMax, pSlow, T, seed, positions, velocities, spaceTime,
   t, i, sortedIdx, leader, gaps, slowMask},

  L = OptionValue["roadLength"];
  n = OptionValue["nVehicles"];
  vMax = OptionValue["vMax"];
  pSlow = OptionValue["pSlow"];
  T = OptionValue["T"];
  seed = OptionValue["seed"];

  SeedRandom[seed];

  positions = Sort @ RandomSample[Range[0, L - 1], n];
  velocities = RandomInteger[{0, vMax}, n];

  spaceTime = ConstantArray[-1, {T, L}];

  Do[
    (* record *)
    Do[
      spaceTime[[t, positions[[i]] + 1]] = velocities[[i]],
      {i, 1, n}
    ];

    (* 1. accelerate *)
    velocities = Clip[velocities + 1, {0, vMax}];

    (* Sort vehicles by position to compute forward gaps *)
    sortedIdx = Ordering[positions];
    positions = positions[[sortedIdx]];
    velocities = velocities[[sortedIdx]];

    (* 2. brake: gap to leader on periodic lattice *)
    gaps = Mod[RotateLeft[positions] - positions - 1, L];
    velocities = MapThread[Min, {velocities, gaps}];

    (* 3. randomisation *)
    slowMask = RandomReal[{0, 1}, n];
    velocities = MapThread[
      If[#2 < pSlow, Max[#1 - 1, 0], #1] &,
      {velocities, slowMask}
    ];

    (* 4. movement *)
    positions = Mod[positions + velocities, L],

    {t, 1, T}
  ];

  <|
    "spacetime" -> spaceTime,
    "roadLength" -> L,
    "nVehicles" -> n,
    "vMax" -> vMax,
    "T" -> T
  |>
];

PlotNaSch[res_Association] := Module[{st, display},
  st = res["spacetime"];
  (* Map empty (-1) to White, speeds to RdYlGn gradient *)
  display = ArrayPlot[
    Reverse @ st,
    DataReversed -> True,
    ColorRules -> {-1 -> White},
    ColorFunction -> (If[# < 0, White,
       ColorData["TemperatureMap"][1 - #/res["vMax"]]] &),
    ColorFunctionScaling -> False,
    Frame -> True,
    FrameLabel -> {"Cell position", "Time step"},
    PlotLabel -> Row[{"Nagel-Schreckenberg: Space-Time (N=", res["nVehicles"],
                       ", L=", res["roadLength"], ")"}],
    ImageSize -> 900,
    AspectRatio -> 0.7
  ];
  display
];

PlotNaSch[res_Association, path_String] := Module[{g = PlotNaSch[res]},
  Export[path, g, ImageResolution -> 150]; path];

End[];
EndPackage[];
