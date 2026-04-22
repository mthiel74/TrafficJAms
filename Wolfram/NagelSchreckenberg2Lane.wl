(* ::Package:: *)

(* NagelSchreckenberg2Lane.wl -- two-lane NaSch cellular automaton with
   symmetric lane-changing rules. Port of
   trafficjams/nagel_schreckenberg_2lane.py. *)

BeginPackage["NagelSchreckenberg2Lane`"];

SimulateNaSch2::usage = "SimulateNaSch2[opts] runs the 2-lane NaSch CA.";
PlotNaSch2::usage = "PlotNaSch2[res] renders the two lane space-time diagrams and the lane-change time series.";
AnimateNaSch2::usage = "AnimateNaSch2[res, path] exports an animated GIF showing both lanes over time.";

Begin["`Private`"];

(* Executes the four NaSch rules on a single lane (position/velocity lists
   already sorted by position). Returns updated {pos, vel}. *)
naschStep[posIn_, velIn_, L_, vMax_, pSlow_] := Module[
  {pos = posIn, vel = velIn, n = Length[posIn], gaps, slow},
  If[n == 0, Return[{pos, vel}]];
  vel = Clip[vel + 1, {0, vMax}];
  gaps = Mod[RotateLeft[pos] - pos - 1, L];
  vel = MapThread[Min, {vel, gaps}];
  slow = RandomReal[{0, 1}, n];
  vel = MapThread[If[#2 < pSlow, Max[#1 - 1, 0], #1] &, {vel, slow}];
  pos = Mod[pos + vel, L];
  {pos, vel}
];

Options[SimulateNaSch2] = {
  "roadLength" -> 500, "nVehicles" -> 150,
  "vMax" -> 5, "pSlow" -> 0.3, "T" -> 300, "seed" -> 42
};

SimulateNaSch2[OptionsPattern[]] := Module[
  {L, nTot, vMax, pSlow, T, seed, nPerLane, pos1, vel1, pos2, vel2,
   spacetime, laneChanges, t, i, n1, n2, sortedIdx, leaderIdx, gapCurrent,
   p, d, gapAhead, gapBehind, otherSet, toSwitch, changes, step,
   curPos, curVel, othPos, othVel, keepMask, order, newPosSelf, newVelSelf,
   newPosOther, newVelOther, stepResult},

  L = OptionValue["roadLength"];
  nTot = OptionValue["nVehicles"];
  vMax = OptionValue["vMax"];
  pSlow = OptionValue["pSlow"];
  T = OptionValue["T"];
  seed = OptionValue["seed"];
  SeedRandom[seed];

  nPerLane = Quotient[nTot, 2];

  pos1 = Sort @ RandomSample[Range[0, L - 1], nPerLane];
  vel1 = RandomInteger[{0, vMax}, nPerLane];
  pos2 = Sort @ RandomSample[Range[0, L - 1], nPerLane];
  vel2 = RandomInteger[{0, vMax}, nPerLane];

  spacetime = ConstantArray[-1, {T, 2, L}];
  laneChanges = ConstantArray[0, T];

  (* One symmetric lane-change sweep for "self"-lane toward "other"-lane.
     Returns updated {selfPos, selfVel, otherPos, otherVel, nSwitches}. *)
  step[curPos_, curVel_, othPos_, othVel_] := Module[
    {n = Length[curPos], order, gaps, othSet, toSwitch2, keep, newSelfP,
     newSelfV, newOthP, newOthV, pi, vi, gapA, gapB, dd},
    order = Ordering[curPos];
    gaps = ConstantArray[0, n];
    (* forward gap in current lane *)
    Do[
      With[{a = order[[k]], b = order[[Mod[k, n] + 1]]},
        gaps[[a]] = Mod[curPos[[b]] - curPos[[a]] - 1, L]
      ],
      {k, n}
    ];
    othSet = Dispatch[Thread[othPos -> True]];
    toSwitch2 = {};
    Do[
      pi = curPos[[i]]; vi = curVel[[i]];
      If[gaps[[i]] >= vi + 1, Continue[]];
      gapA = 0;
      Do[
        If[(Mod[pi + dd, L] /. othSet) === True, Break[]];
        gapA++,
        {dd, 1, vMax + 1}
      ];
      If[gapA <= vi, Continue[]];
      gapB = 0;
      Do[
        If[(Mod[pi - dd, L] /. othSet) === True, Break[]];
        gapB++,
        {dd, 1, vMax + 1}
      ];
      If[gapB <= vMax, Continue[]];
      If[!((pi /. othSet) === True),
        AppendTo[toSwitch2, i];
        (* update otherSet so subsequent vehicles can't pick the same target cell *)
        othSet = Dispatch[Append[Normal[othSet], pi -> True]];
      ],
      {i, n}
    ];
    keep = Complement[Range[n], toSwitch2];
    newSelfP = curPos[[keep]]; newSelfV = curVel[[keep]];
    newOthP = Join[othPos, curPos[[toSwitch2]]];
    newOthV = Join[othVel, curVel[[toSwitch2]]];
    {newSelfP, newSelfV, newOthP, newOthV, Length[toSwitch2]}
  ];

  Do[
    (* record state *)
    Do[
      spacetime[[t, 1, pos1[[i]] + 1]] = vel1[[i]],
      {i, Length[pos1]}
    ];
    Do[
      spacetime[[t, 2, pos2[[i]] + 1]] = vel2[[i]],
      {i, Length[pos2]}
    ];

    changes = 0;
    stepResult = step[pos1, vel1, pos2, vel2];
    pos1 = stepResult[[1]]; vel1 = stepResult[[2]];
    pos2 = stepResult[[3]]; vel2 = stepResult[[4]];
    changes += stepResult[[5]];

    stepResult = step[pos2, vel2, pos1, vel1];
    pos2 = stepResult[[1]]; vel2 = stepResult[[2]];
    pos1 = stepResult[[3]]; vel1 = stepResult[[4]];
    changes += stepResult[[5]];

    laneChanges[[t]] = changes;

    (* NaSch update per lane (requires sorted positions) *)
    order = Ordering[pos1]; pos1 = pos1[[order]]; vel1 = vel1[[order]];
    With[{r = naschStep[pos1, vel1, L, vMax, pSlow]}, pos1 = r[[1]]; vel1 = r[[2]]];

    order = Ordering[pos2]; pos2 = pos2[[order]]; vel2 = vel2[[order]];
    With[{r = naschStep[pos2, vel2, L, vMax, pSlow]}, pos2 = r[[1]]; vel2 = r[[2]]],

    {t, 1, T}
  ];

  <|
    "spacetime" -> spacetime,
    "roadLength" -> L,
    "nVehicles" -> nTot,
    "vMax" -> vMax,
    "T" -> T,
    "laneChanges" -> laneChanges
  |>
];

PlotNaSch2[res_Association] := Module[
  {vMax, lanePlot, laneChangePlot, cmap},
  vMax = res["vMax"];
  cmap = If[# < 0, White, ColorData["TemperatureMap"][1 - #/vMax]] &;
  lanePlot[laneIdx_] := ArrayPlot[
    Reverse @ res["spacetime"][[All, laneIdx]],
    DataReversed -> True,
    ColorFunction -> cmap,
    ColorFunctionScaling -> False,
    Frame -> True,
    FrameLabel -> {"Cell position", "Time step"},
    PlotLabel -> Row[{"Lane ", laneIdx}],
    ImageSize -> 440, AspectRatio -> 0.8
  ];
  laneChangePlot = ListLinePlot[
    Transpose[{res["laneChanges"], Range[res["T"]]}],
    Frame -> True,
    FrameLabel -> {"Lane changes", "Time step"},
    PlotLabel -> "Lane changes / step",
    PlotStyle -> Blue,
    ImageSize -> 260, AspectRatio -> 1.3
  ];
  Column[{
    Style[Row[{"2-lane NaSch (N=", res["nVehicles"],
               ", L=", res["roadLength"], ")"}], 14, Bold],
    GraphicsRow[{lanePlot[1], lanePlot[2], laneChangePlot},
                ImageSize -> 1150, Spacings -> 20]
  }]
];

PlotNaSch2[res_Association, path_String] := Module[{g = PlotNaSch2[res]},
  Export[path, g, ImageResolution -> 150]; path];

(* -------- Animation -------- *)

speedColour[v_, vMax_] := If[v < 0, White,
  Blend[{RGBColor[0.85, 0.1, 0.1], RGBColor[1.0, 0.85, 0.2],
         RGBColor[0.1, 0.7, 0.2]}, Clip[v/vMax, {0, 1}]]];

nasch2Frame[res_Association, t_Integer] := Module[
  {L = res["roadLength"], vMax = res["vMax"], lanes, positions, speeds, i, st},
  st = res["spacetime"][[t]];
  Graphics[
    {
      (* road backgrounds *)
      {GrayLevel[0.85], Rectangle[{0, 1.2}, {L, 1.9}]},  (* lane 1 *)
      {GrayLevel[0.85], Rectangle[{0, 0.2}, {L, 0.9}]},  (* lane 2 *)
      {GrayLevel[0.4], Thickness[0.002], Line[{{0, 1.05}, {L, 1.05}}]},

      (* lane 1 cars *)
      Table[
        If[st[[1, i]] >= 0,
          {speedColour[st[[1, i]], vMax], EdgeForm[Darker[Gray]],
            Disk[{i - 1, 1.55}, 2.5]},
          Nothing
        ],
        {i, L}
      ],
      (* lane 2 cars *)
      Table[
        If[st[[2, i]] >= 0,
          {speedColour[st[[2, i]], vMax], EdgeForm[Darker[Gray]],
            Disk[{i - 1, 0.55}, 2.5]},
          Nothing
        ],
        {i, L}
      ],

      (* labels *)
      Text[Style["Lane 1", 10, Darker[Gray]], {-15, 1.55}, {1, 0}],
      Text[Style["Lane 2", 10, Darker[Gray]], {-15, 0.55}, {1, 0}]
    },
    PlotRange -> {{-20, L + 5}, {0, 2.3}},
    AspectRatio -> 0.12,
    Background -> GrayLevel[0.97],
    ImageSize -> 700,
    PlotLabel -> Row[{"2-lane NaSch  t = ", t, "  (lane changes so far = ",
                      Total[Take[res["laneChanges"], t]], ")"}]
  ]
];

Options[AnimateNaSch2] = {"frameRate" -> 12, "stride" -> 2, "maxSteps" -> 120};

AnimateNaSch2[res_Association, path_String, OptionsPattern[]] := Module[
  {stride, fr, maxSteps, idx, images},
  stride = OptionValue["stride"];
  fr = OptionValue["frameRate"];
  maxSteps = Min[OptionValue["maxSteps"], res["T"]];
  idx = Range[1, maxSteps, stride];
  images = nasch2Frame[res, #] & /@ idx;
  Export[path, images,
    "DisplayDurations" -> ConstantArray[1.0/fr, Length[images]]];
  path
];

End[];
EndPackage[];
