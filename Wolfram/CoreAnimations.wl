(* ::Package:: *)

(* CoreAnimations.wl -- GIF animation helpers for the core single-road
   traffic models (IDM ring, Bando ring, NaSch highway, LWR corridor,
   Payne-Whitham corridor). These consume the result Associations
   returned by Simulate<Model>[] in the corresponding model files and
   export an animated GIF. All renderers use the same speed colour
   palette (red = stopped, yellow = slowing, green = free flow). *)

BeginPackage["CoreAnimations`"];

AnimateIDMRing::usage = "AnimateIDMRing[idmResult, path, opts] exports a ring-road animation of the IDM run.";
AnimateBandoRing::usage = "AnimateBandoRing[bandoResult, path, opts] exports a ring-road animation of the Bando OVM run.";
AnimateNaSchHighway::usage = "AnimateNaSchHighway[naschResult, path, opts] exports a horizontal-highway animation with the growing space-time panel.";
AnimateLWRCorridor::usage = "AnimateLWRCorridor[lwrResult, path, opts] exports a density-over-corridor animation.";
AnimatePWCorridor::usage = "AnimatePWCorridor[pwResult, path, opts] exports a paired density/velocity corridor animation.";

Begin["`Private`"];

speedCol[v_, vmax_] := Blend[
  {RGBColor[0.85, 0.1, 0.1], RGBColor[1.0, 0.85, 0.2], RGBColor[0.15, 0.7, 0.2]},
  Clip[v/vmax, {0, 1}]
];

(* --------- IDM ring --------- *)

idmRingFrame[res_Association, k_Integer, vmax_] := Module[
  {L = res["roadLength"], n = res["nVehicles"], pos, vel, t, angles, disks,
   diskR},
  pos = res["positions"][[k]];
  vel = res["velocities"][[k]];
  t = res["times"][[k]];
  angles = 2 Pi pos/L;
  diskR = Clip[0.9 Pi/n, {0.03, 0.1}];  (* scale radius so cars don't overlap *)
  disks = MapThread[
    {speedCol[#2, vmax], EdgeForm[Directive[Thin, GrayLevel[0.3]]],
      Disk[{Cos[#1], Sin[#1]}, diskR]} &,
    {angles, vel}
  ];
  Graphics[
    {
      {GrayLevel[0.82], Thickness[0.03], Circle[{0, 0}, 1]},
      disks,
      Text[Style[Row[{"t = ", NumberForm[N[t], {4, 1}], " s"}], 12],
        {0, 0.05}],
      Text[Style[Row[{"\[LeftAngleBracket]v\[RightAngleBracket] = ",
        NumberForm[N[Mean[vel]], {4, 2}], " m/s   spread [",
        NumberForm[N[Min[vel]], {4, 1}], ", ",
        NumberForm[N[Max[vel]], {4, 1}], "]"}], 10, Darker[Gray]],
        {0, -0.12}]
    },
    PlotRange -> 1.3 {{-1, 1}, {-1, 1}},
    AspectRatio -> 1,
    Background -> GrayLevel[0.97],
    ImageSize -> 480,
    PlotLabel -> Row[{"IDM ring road, ", n, " vehicles"}]
  ]
];

Options[AnimateIDMRing] = {"vmax" -> 15, "frameStep" -> 1,
  "displayDuration" -> 0.15};

AnimateIDMRing[res_Association, path_String, OptionsPattern[]] := Module[
  {vmax, step, dispDur, nFrames, idx, images},
  vmax = OptionValue["vmax"];
  step = OptionValue["frameStep"];
  dispDur = OptionValue["displayDuration"];
  nFrames = Length[res["times"]];
  idx = Range[1, nFrames, step];
  images = idmRingFrame[res, #, vmax] & /@ idx;
  Export[path, images,
    "DisplayDurations" -> ConstantArray[dispDur, Length[images]],
    "AnimationRepetitions" -> Infinity];
  path
];

(* --------- Bando ring --------- *)

bandoRingFrame[res_Association, k_Integer, vmax_] := Module[
  {L = res["roadLength"], n = res["nVehicles"], pos, vel, t, angles, disks,
   meanSpd, diskR},
  pos = res["positions"][[k]];
  vel = res["velocities"][[k]];
  t = res["times"][[k]];
  meanSpd = Mean[vel];
  angles = 2 Pi pos/L;
  diskR = Clip[0.9 Pi/n, {0.04, 0.11}];
  disks = MapThread[
    {speedCol[#2, vmax], EdgeForm[Directive[Thin, GrayLevel[0.3]]],
      Disk[{Cos[#1], Sin[#1]}, diskR]} &,
    {angles, vel}
  ];
  Graphics[
    {
      {GrayLevel[0.82], Thickness[0.03], Circle[{0, 0}, 1]},
      disks,
      Text[Style[Row[{"\[LeftAngleBracket]v\[RightAngleBracket] = ",
        NumberForm[N[meanSpd], {4, 2}], " m/s   spread [",
        NumberForm[N[Min[vel]], {4, 1}], ", ",
        NumberForm[N[Max[vel]], {4, 1}], "]"}], 11], {0, 0.06}],
      Text[Style[Row[{"t = ", NumberForm[N[t], {4, 1}], " s"}], 10,
        Gray], {0, -0.06}]
    },
    PlotRange -> 1.3 {{-1, 1}, {-1, 1}},
    AspectRatio -> 1,
    Background -> GrayLevel[0.97],
    ImageSize -> 480,
    PlotLabel -> Row[{"Bando OVM ring, \[Kappa] = ", res["kappa"],
                       ", ", n, " vehicles"}]
  ]
];

Options[AnimateBandoRing] = {"vmax" -> 15, "frameStep" -> 1,
  "displayDuration" -> 0.15};

AnimateBandoRing[res_Association, path_String, OptionsPattern[]] := Module[
  {vmax, step, dispDur, nFrames, idx, images},
  vmax = OptionValue["vmax"];
  step = OptionValue["frameStep"];
  dispDur = OptionValue["displayDuration"];
  nFrames = Length[res["times"]];
  idx = Range[1, nFrames, step];
  images = bandoRingFrame[res, #, vmax] & /@ idx;
  Export[path, images,
    "DisplayDurations" -> ConstantArray[dispDur, Length[images]],
    "AnimationRepetitions" -> Infinity];
  path
];

(* --------- NaSch single lane horizontal highway ---------
   Draws the road as a horizontal strip with cars as coloured cell-width
   rectangles, plus the accumulating space-time diagram as a raster
   below. Everything goes into a single Graphics so the ImageSize /
   AspectRatio aren't fought over by a Column. *)

(* Layout is fixed up front so every frame has the same size.
   stTopFixed = height reserved for the space-time panel, same for all t;
   partial space-times are padded with white above the current row. *)

naschFrame[res_Association, t_Integer] := Module[
  {L = res["roadLength"], vMax = res["vMax"], st, Tfull = res["T"],
   win = Min[300, res["roadLength"]], i, subst, stHeight, roadY, stTop, totalH,
   padRows, paddedRaster},
  st = res["spacetime"][[t]];
  stHeight = Min[win*0.9, Tfull*1.0];  (* fixed across all frames *)
  stTop = stHeight;
  roadY = stTop + 4;
  totalH = roadY + 4;
  (* build a raster the size of the FULL accumulated space-time,
     but fill rows beyond t with white *)
  subst = Join[
    res["spacetime"][[1 ;; t, 1 ;; win]],
    ConstantArray[-1, {Tfull - t, win}]
  ];
  paddedRaster = Map[
    If[# < 0, {1, 1, 1}, List @@ speedCol[#, vMax]] &,
    subst, {2}
  ];
  Graphics[
    {
      (* space-time raster -- rows 1..t contain data, rows t+1..Tfull white *)
      Raster[Reverse @ paddedRaster, {{0, 0}, {win, stTop}}],
      (* frame *)
      {GrayLevel[0.3], Thickness[0.0015],
        Line[{{0, 0}, {win, 0}, {win, stTop}, {0, stTop}, {0, 0}}]},
      Text[Style[Row[{"Space-time diagram (step ", t, " of ", Tfull, ")"}],
        10, Gray], {win/2, stTop + 1}, {0, 0}],

      (* road background *)
      {GrayLevel[0.82], Rectangle[{0, roadY}, {win, roadY + 3}]},
      (* cars as rectangles *)
      Table[
        If[st[[i]] >= 0,
          {speedCol[st[[i]], vMax],
            EdgeForm[Directive[Thin, GrayLevel[0.35]]],
            Rectangle[{i - 1 + 0.1, roadY + 0.3},
                      {i - 0.1, roadY + 2.7}]},
          Nothing
        ],
        {i, win}
      ]
    },
    PlotRange -> {{-5, win + 5}, {-2, totalH + 1}},
    AspectRatio -> totalH/(win + 10),
    ImageSize -> 640,
    Background -> GrayLevel[0.98],
    PlotLabel -> Row[{"NaSch highway (first ", win, " of ", L, " cells)"}]
  ]
];

Options[AnimateNaSchHighway] = {"frameStep" -> 4, "displayDuration" -> 0.2,
  "maxFrames" -> 80};

AnimateNaSchHighway[res_Association, path_String, OptionsPattern[]] := Module[
  {step, dispDur, maxFr, nSteps, stride, idx, images},
  step = OptionValue["frameStep"];
  dispDur = OptionValue["displayDuration"];
  maxFr = OptionValue["maxFrames"];
  nSteps = res["T"];
  stride = Max[step, Ceiling[nSteps/maxFr]];
  idx = Range[1, nSteps, stride];
  images = naschFrame[res, #] & /@ idx;
  Export[path, images,
    "DisplayDurations" -> ConstantArray[dispDur, Length[images]],
    "AnimationRepetitions" -> Infinity];
  path
];

(* --------- LWR corridor --------- *)

lwrFrame[res_Association, k_Integer] := Module[
  {x = res["x"], rho, t, rhoMax = res["rhoMax"], vMax = res["vMax"], rhoCrit,
   bands},
  rhoCrit = rhoMax/2;
  rho = res["density"][[k]];
  t = res["t"][[k]];
  (* colour each cell by the local ratio rho/rhoMax: free flow green, jam red *)
  bands = MapThread[
    {speedCol[Max[vMax (1 - #2/rhoMax), 0.0], vMax],
     Rectangle[{#1 - 0.02, 0}, {#1 + 0.02, 8}]} &,
    {x, rho}
  ];
  Column[{
    ListLinePlot[Transpose[{x, rho}],
      PlotRange -> {{First[x], Last[x]}, {0, rhoMax 1.05}},
      Filling -> Bottom, FillingStyle -> Directive[Red, Opacity[0.25]],
      PlotStyle -> Directive[Red, Thickness[0.008]],
      Frame -> True,
      FrameLabel -> {"Position (km)", "Density (veh/km)"},
      PlotLabel -> Row[{"LWR corridor  t = ",
        NumberForm[N[t], {4, 3}], " h"}],
      ImageSize -> 520, AspectRatio -> 0.5],
    Graphics[bands,
      PlotRange -> {{First[x], Last[x]}, {0, 8}},
      AspectRatio -> 8.0/(Last[x] - First[x]),
      ImageSize -> 520,
      Background -> White,
      PlotLabel -> "Road congestion (green=free flow, red=jam)"]
  }, Alignment -> Center]
];

Options[AnimateLWRCorridor] = {"frameStep" -> 1, "displayDuration" -> 0.18};

AnimateLWRCorridor[res_Association, path_String, OptionsPattern[]] := Module[
  {step, dispDur, nFrames, idx, images},
  step = OptionValue["frameStep"];
  dispDur = OptionValue["displayDuration"];
  nFrames = Length[res["t"]];
  idx = Range[1, nFrames, step];
  images = lwrFrame[res, #] & /@ idx;
  Export[path, images,
    "DisplayDurations" -> ConstantArray[dispDur, Length[images]],
    "AnimationRepetitions" -> Infinity];
  path
];

(* --------- Payne-Whitham corridor --------- *)

pwFrame[res_Association, k_Integer] := Module[
  {x = res["x"], rho, v, rhoMax = res["rhoMax"], vMax = res["vMax"], t},
  rho = res["density"][[k]];
  v = res["velocity"][[k]];
  t = res["t"][[k]];
  GraphicsRow[{
    ListLinePlot[Transpose[{x, rho}],
      Filling -> Bottom, FillingStyle -> Directive[Red, Opacity[0.25]],
      PlotStyle -> Directive[Red, Thickness[0.008]],
      PlotRange -> {All, {0, rhoMax 1.05}},
      Frame -> True,
      FrameLabel -> {"Position (km)", "Density (veh/km)"},
      PlotLabel -> Row[{"Density  t = ", NumberForm[N[t], {4, 4}], " h"}],
      ImageSize -> 380, AspectRatio -> 0.7],
    ListLinePlot[Transpose[{x, v}],
      Filling -> Bottom, FillingStyle -> Directive[Green, Opacity[0.25]],
      PlotStyle -> Directive[Darker[Green], Thickness[0.008]],
      PlotRange -> {All, {0, vMax 1.05}},
      Frame -> True,
      FrameLabel -> {"Position (km)", "Velocity (km/h)"},
      PlotLabel -> "Velocity",
      ImageSize -> 380, AspectRatio -> 0.7]
  }, ImageSize -> 780]
];

Options[AnimatePWCorridor] = {"frameStep" -> 1, "displayDuration" -> 0.18};

AnimatePWCorridor[res_Association, path_String, OptionsPattern[]] := Module[
  {step, dispDur, nFrames, idx, images},
  step = OptionValue["frameStep"];
  dispDur = OptionValue["displayDuration"];
  nFrames = Length[res["t"]];
  idx = Range[1, nFrames, step];
  images = pwFrame[res, #] & /@ idx;
  Export[path, images,
    "DisplayDurations" -> ConstantArray[dispDur, Length[images]],
    "AnimationRepetitions" -> Infinity];
  path
];

End[];
EndPackage[];
