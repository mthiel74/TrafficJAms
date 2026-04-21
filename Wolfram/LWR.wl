(* ::Package:: *)

(* LWR.wl -- Lighthill-Whitham-Richards model with Godunov scheme.
   Port of trafficjams/lwr.py. Greenshields fundamental diagram:
     Q(rho) = rho * v_max * (1 - rho/rho_max)
*)

BeginPackage["LWR`"];

Greenshields::usage = "Greenshields[rho, rhoMax, vMax] returns flow Q(rho).";
GodunovFlux::usage = "GodunovFlux[rhoL, rhoR, rhoMax, vMax] returns the Godunov numerical flux.";
SimulateLWR::usage = "SimulateLWR[opts] runs the LWR model on a corridor.";
PlotLWR::usage = "PlotLWR[res] draws the density-evolution plot.";
AnimateLWR::usage = "AnimateLWR[res, path] exports an animated density profile + road-strip GIF.";

Begin["`Private`"];

Greenshields[rho_, rhoMax_, vMax_] := rho vMax (1 - rho/rhoMax);

GodunovFlux[rhoL_, rhoR_, rhoMax_, vMax_] := Module[{rhoCrit, qL, qR, qCrit},
  rhoCrit = rhoMax/2;
  qL = Greenshields[rhoL, rhoMax, vMax];
  qR = Greenshields[rhoR, rhoMax, vMax];
  qCrit = Greenshields[rhoCrit, rhoMax, vMax];
  Which[
    rhoL <= rhoR,
      Which[
        rhoL >= rhoCrit, qL,
        rhoR <= rhoCrit, qR,
        True, qCrit
      ],
    True,  (* rhoL > rhoR *)
      If[qL <= qR,
        qL,
        If[rhoL >= rhoCrit, qL, Max[qL, qR]]
      ]
  ]
];

Options[SimulateLWR] = {
  "L" -> 10.0, "nx" -> 200, "T" -> 0.5,
  "rhoMax" -> 150.0, "vMax" -> 30.0
};

SimulateLWR[OptionsPattern[]] := Module[
  {L, nx, T, rhoMax, vMax, dx, dt, nt, x, rho, history, times, n, rhoNew, fl, fr, sampleEvery},

  L = OptionValue["L"]; nx = OptionValue["nx"]; T = OptionValue["T"];
  rhoMax = OptionValue["rhoMax"]; vMax = OptionValue["vMax"];

  dx = L/nx;
  dt = 0.5 dx/vMax;   (* CFL *)
  nt = Floor[T/dt];

  x = N @ Subdivide[0., L, nx - 1];
  rho = ConstantArray[20.0, nx];
  Do[If[3 < x[[i]] < 5, rho[[i]] = 120.0], {i, nx}];

  history = {rho}; times = {0.0};
  sampleEvery = Max[1, Quotient[nt, 50]];

  Do[
    rhoNew = rho;
    Do[
      fl = GodunovFlux[rho[[i - 1]], rho[[i]], rhoMax, vMax];
      fr = GodunovFlux[rho[[i]], rho[[i + 1]], rhoMax, vMax];
      rhoNew[[i]] = rho[[i]] - dt/dx (fr - fl),
      {i, 2, nx - 1}
    ];
    rho = Clip[rhoNew, {0, rhoMax}];
    If[Mod[n, sampleEvery] == 0,
      AppendTo[history, rho]; AppendTo[times, n dt];
    ],
    {n, 1, nt}
  ];

  <|
    "x" -> x,
    "t" -> times,
    "density" -> history,
    "rhoMax" -> rhoMax,
    "vMax" -> vMax,
    "dx" -> dx,
    "dt" -> dt
  |>
];

PlotLWR[res_Association] := Module[
  {x, t, density, rhoMax, spacePlot, snapPlot, nSnaps, indices},

  x = res["x"]; t = res["t"]; density = res["density"]; rhoMax = res["rhoMax"];

  spacePlot = ArrayPlot[
    Reverse @ density,
    DataReversed -> True,
    ColorFunction -> "SunsetColors",
    ColorFunctionScaling -> True,
    Frame -> True,
    FrameLabel -> {"Position index (x ~ [0, L])", "Time index"},
    PlotLabel -> "LWR: Density Evolution (A90-like corridor)",
    PlotLegends -> Automatic,
    ImageSize -> 520, AspectRatio -> 0.9
  ];

  nSnaps = Min[5, Length[t]];
  indices = Round @ Subdivide[1, Length[t], nSnaps - 1];
  snapPlot = ListLinePlot[
    Table[Transpose[{x, density[[i]]}], {i, indices}],
    Frame -> True, FrameLabel -> {"Position (km)", "Density (veh/km)"},
    PlotRange -> {All, {0, rhoMax}},
    PlotLabel -> "Density profiles at different times",
    PlotLegends -> (Row[{"t=", NumberForm[t[[#]], {4, 3}], " h"}] & /@ indices),
    ImageSize -> 520, AspectRatio -> 0.9
  ];

  GraphicsRow[{spacePlot, snapPlot}, ImageSize -> 1100]
];

PlotLWR[res_Association, path_String] := Module[{g = PlotLWR[res]},
  Export[path, g, ImageResolution -> 150]; path];

(* Density-profile + road-strip animation. *)
densityColour[rho_, rhoMax_] := Blend[
  {RGBColor[0.15, 0.7, 0.2], RGBColor[1.0, 0.85, 0.2], RGBColor[0.8, 0.1, 0.1]},
  Clip[rho/rhoMax, {0, 1}]
];

lwrFrame[res_Association, k_Integer] := Module[
  {x, rho, rhoMax, L, strip, nx, dx, profile, t, i, rects},
  x = res["x"]; rho = res["density"][[k]]; rhoMax = res["rhoMax"];
  L = Last[x]; nx = Length[x]; dx = res["dx"]; t = res["t"][[k]];

  profile = ListLinePlot[Transpose[{x, rho}],
    Frame -> True, FrameLabel -> {"Position (km)", "Density (veh/km)"},
    PlotRange -> {{0, L}, {0, rhoMax 1.05}},
    PlotStyle -> Directive[Thickness[0.005], Black],
    Filling -> Bottom,
    FillingStyle -> densityColour[#, rhoMax] & /@ rho,
    GridLines -> Automatic, GridLinesStyle -> LightGray,
    PlotLabel -> Row[{"LWR corridor  t = ", NumberForm[N[t], {4, 3}], " h"}],
    ImageSize -> 600, AspectRatio -> 0.35
  ];

  rects = Table[
    {densityColour[rho[[i]], rhoMax],
     Rectangle[{x[[i]] - dx/2, 0}, {x[[i]] + dx/2, 1}]},
    {i, nx}
  ];
  strip = Graphics[
    {rects, {Black, Thickness[0.003],
     Line[{{0, 0}, {L, 0}}], Line[{{0, 1}, {L, 1}}]}},
    PlotRange -> {{0, L}, {0, 1}}, ImageSize -> 600,
    AspectRatio -> 0.08, ImagePadding -> {{45, 10}, {10, 10}},
    PlotLabel -> "Road congestion (green=free flow, red=jam)"
  ];

  Column[{profile, strip}, Spacings -> 0]
];

Options[AnimateLWR] = {"frameStep" -> 1, "displayDuration" -> 0.08};
AnimateLWR[res_Association, path_String, OptionsPattern[]] := Module[
  {frames, step = OptionValue["frameStep"], nFrames = Length[res["t"]]},
  frames = Table[lwrFrame[res, k], {k, 1, nFrames, step}];
  Export[path, frames, "GIF",
    "AnimationRepetitions" -> Infinity,
    "DisplayDurations" -> OptionValue["displayDuration"]];
  path
];

End[];
EndPackage[];
