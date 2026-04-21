(* ::Package:: *)

(* PayneWhitham.wl -- second-order traffic PDE.
   Port of trafficjams/payne_whitham.py. *)

BeginPackage["PayneWhitham`"];

EquilibriumSpeed::usage = "EquilibriumSpeed[rho, rhoMax, vMax] returns Greenshields equilibrium speed Ve(rho).";
SimulatePW::usage = "SimulatePW[opts] runs the Payne-Whitham model on a merge-zone corridor.";
PlotPW::usage = "PlotPW[res] renders density and velocity evolution plots.";
AnimatePW::usage = "AnimatePW[res, path] exports an animated density/velocity profile GIF.";

Begin["`Private`"];

EquilibriumSpeed[rho_, rhoMax_, vMax_] := vMax (1 - rho/rhoMax);

Options[SimulatePW] = {
  "L" -> 10.0, "nx" -> 200, "T" -> 0.3,
  "rhoMax" -> 150.0, "vMax" -> 30.0, "tau" -> 0.01, "c0" -> 5.0
};

SimulatePW[OptionsPattern[]] := Module[
  {L, nx, T, rhoMax, vMax, tau, c0, dx, dt, nt, x, rho, v, rhoHist, vHist,
   times, sampleEvery, step, rhoNew, vNew, i, fluxR, fluxL, convection,
   relaxation, pressure},

  L = OptionValue["L"]; nx = OptionValue["nx"]; T = OptionValue["T"];
  rhoMax = OptionValue["rhoMax"]; vMax = OptionValue["vMax"];
  tau = OptionValue["tau"]; c0 = OptionValue["c0"];

  dx = L/nx;
  dt = 0.3 dx/(vMax + c0);
  nt = Floor[T/dt];

  x = N @ Subdivide[0., L, nx - 1];
  rho = ConstantArray[40.0, nx];
  Do[If[4 < x[[i]] < 6, rho[[i]] = 90.0], {i, nx}];
  v = EquilibriumSpeed[rho, rhoMax, vMax];

  rhoHist = {rho}; vHist = {v}; times = {0.0};
  sampleEvery = Max[1, Quotient[nt, 50]];

  Do[
    rhoNew = rho; vNew = v;
    Do[
      fluxR = rho[[i]] v[[i]];
      fluxL = rho[[i - 1]] v[[i - 1]];
      rhoNew[[i]] = rho[[i]] - dt/dx (fluxR - fluxL);
      If[rho[[i]] > 1.*^-6,
        convection = -v[[i]] (v[[i]] - v[[i - 1]])/dx;
        relaxation = (EquilibriumSpeed[rho[[i]], rhoMax, vMax] - v[[i]])/tau;
        pressure = -c0^2/rho[[i]] (rho[[i + 1]] - rho[[i]])/dx;
        vNew[[i]] = v[[i]] + dt (convection + relaxation + pressure);
      ],
      {i, 2, nx - 1}
    ];
    rho = Clip[rhoNew, {0.1, rhoMax}];
    v = Clip[vNew, {0, vMax}];
    If[Mod[step, sampleEvery] == 0,
      AppendTo[rhoHist, rho]; AppendTo[vHist, v]; AppendTo[times, step dt];
    ],
    {step, 1, nt}
  ];

  <|
    "x" -> x, "t" -> times,
    "density" -> rhoHist, "velocity" -> vHist,
    "rhoMax" -> rhoMax, "vMax" -> vMax
  |>
];

PlotPW[res_Association] := Module[{rhoPlot, vPlot},
  rhoPlot = ArrayPlot[
    Reverse @ res["density"],
    DataReversed -> True,
    ColorFunction -> "SunsetColors", ColorFunctionScaling -> True,
    Frame -> True,
    FrameLabel -> {"Position index", "Time index"},
    PlotLabel -> "Payne-Whitham: Density (merge zone)",
    PlotLegends -> Automatic,
    ImageSize -> 520, AspectRatio -> 0.9
  ];
  vPlot = ArrayPlot[
    Reverse @ res["velocity"],
    DataReversed -> True,
    ColorFunction -> "TemperatureMap", ColorFunctionScaling -> True,
    Frame -> True,
    FrameLabel -> {"Position index", "Time index"},
    PlotLabel -> "Payne-Whitham: Velocity",
    PlotLegends -> Automatic,
    ImageSize -> 520, AspectRatio -> 0.9
  ];
  GraphicsRow[{rhoPlot, vPlot}, ImageSize -> 1100]
];

PlotPW[res_Association, path_String] := Module[{g = PlotPW[res]},
  Export[path, g, ImageResolution -> 150]; path];

pwFrame[res_Association, k_Integer] := Module[
  {x, rho, v, rhoMax, vMax, L, t, rhoPlot, vPlot},
  x = res["x"]; rho = res["density"][[k]]; v = res["velocity"][[k]];
  rhoMax = res["rhoMax"]; vMax = res["vMax"];
  L = Last[x]; t = res["t"][[k]];

  rhoPlot = ListLinePlot[Transpose[{x, rho}],
    Frame -> True, FrameLabel -> {"Position (km)", "Density (veh/km)"},
    PlotRange -> {{0, L}, {0, rhoMax}}, PlotStyle -> Red,
    Filling -> Bottom, FillingStyle -> Directive[Red, Opacity[0.25]],
    PlotLabel -> Row[{"Density  t = ", NumberForm[N[t], {5, 4}], " h"}],
    GridLines -> Automatic, GridLinesStyle -> LightGray,
    ImageSize -> 450, AspectRatio -> 0.6
  ];
  vPlot = ListLinePlot[Transpose[{x, v}],
    Frame -> True, FrameLabel -> {"Position (km)", "Velocity (km/h)"},
    PlotRange -> {{0, L}, {0, vMax}}, PlotStyle -> Darker[Green],
    Filling -> Bottom, FillingStyle -> Directive[Green, Opacity[0.25]],
    PlotLabel -> "Velocity",
    GridLines -> Automatic, GridLinesStyle -> LightGray,
    ImageSize -> 450, AspectRatio -> 0.6
  ];
  GraphicsRow[{rhoPlot, vPlot}, ImageSize -> 900]
];

Options[AnimatePW] = {"frameStep" -> 1, "displayDuration" -> 0.08};
AnimatePW[res_Association, path_String, OptionsPattern[]] := Module[
  {frames, step = OptionValue["frameStep"], nFrames = Length[res["t"]]},
  frames = Table[pwFrame[res, k], {k, 1, nFrames, step}];
  Export[path, frames, "GIF",
    "AnimationRepetitions" -> Infinity,
    "DisplayDurations" -> OptionValue["displayDuration"]];
  path
];

End[];
EndPackage[];
