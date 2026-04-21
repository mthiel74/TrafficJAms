(* ::Package:: *)

(* NagelSchreckenberg2Lane.wl -- two-lane NaSch cellular automaton with
   symmetric lane-changing rules. Port of
   trafficjams/nagel_schreckenberg_2lane.py. *)

BeginPackage["NagelSchreckenberg2Lane`"];

SimulateNaSch2::usage = "SimulateNaSch2[opts] runs the 2-lane NaSch CA.";
PlotNaSch2::usage = "PlotNaSch2[res] renders the two lane space-time diagrams and the lane-change time series.";

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

End[];
EndPackage[];
