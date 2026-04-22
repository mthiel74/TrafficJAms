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

End[];
EndPackage[];
