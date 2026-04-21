(* ::Package:: *)

(* IntersectionControl.wl -- junction classification, fixed-time signal
   controller, roundabout gap acceptance, and a virtual-gap delay model
   for uncontrolled junctions. Port of
   trafficjams/intersection_control.py.

   Consumed by the full Aberdeen multi-agent simulation. *)

BeginPackage["IntersectionControl`"];

ClassifyNodes::usage = "ClassifyNodes[nodeTags, edgeJunctionTags] classifies each OSM node as signal/roundabout/stop/giveWay/uncontrolled.";
MakeSignalController::usage = "MakeSignalController[nodeId, incomingEdges, cycle, amber, offset] returns an Association describing a round-robin signal phase plan.";
SignalIsGreen::usage = "SignalIsGreen[controller, edge, simTime] returns True if the edge has green at simTime.";
BuildSignalControllers::usage = "BuildSignalControllers[incomingEdgesAt, nodeClass, cycle, seed] returns one controller per 'signal' node.";
RoundaboutCanEnter::usage = "RoundaboutCanEnter[approachEdge, circulatingOcc, leaderSpeeds, edgeLengths, criticalHeadway] applies the TRL 281 gap-acceptance rule.";
IntersectionVirtualGap::usage = "IntersectionVirtualGap[nodeDegree, nodeClass] returns a virtual IDM gap (m) that produces realistic intersection delay.";
SimulateSignalDemo::usage = "SimulateSignalDemo[opts] builds a schematic 4-arm controller and returns its phase timetable.";
PlotSignalDemo::usage = "PlotSignalDemo[res] draws a Gantt-style phase chart over two cycles.";

Begin["`Private`"];

(* -------- node classification -------- *)

ClassifyNodes[nodeTags_Association, edgeJunctionTags_Association] := Module[
  {roundaboutNodes, cls, node, hw, edge, junc},
  roundaboutNodes = <||>;
  KeyValueMap[
    Function[{edge, junc},
      If[junc === "roundabout",
        roundaboutNodes[edge[[1]]] = True;
        roundaboutNodes[edge[[2]]] = True;
      ]
    ],
    edgeJunctionTags
  ];
  cls = <||>;
  KeyValueMap[
    Function[{node, hw},
      cls[node] = Which[
        hw === "traffic_signals", "signal",
        hw === "mini_roundabout" || KeyExistsQ[roundaboutNodes, node], "roundabout",
        hw === "stop", "stop",
        MemberQ[{"give_way", "yield"}, hw], "giveWay",
        True, "uncontrolled"
      ]
    ],
    nodeTags
  ];
  cls
];

(* -------- signal controller -------- *)

MakeSignalController[nodeId_, incomingEdges_List, cycle_: 90.0,
                     amber_: 4.0, offset_: 0.0] := Module[
  {phases, nPhases, lost, usable, greenTimes, starts, t},
  phases = If[Length[incomingEdges] > 2,
    List /@ incomingEdges,
    {incomingEdges}
  ];
  nPhases = Length[phases];
  lost = amber nPhases;
  usable = Max[cycle - lost, 10.0 nPhases];
  greenTimes = ConstantArray[usable/nPhases, nPhases];
  starts = Accumulate @ Prepend[Most[greenTimes + amber], 0.0];
  <|
    "nodeId" -> nodeId,
    "cycle" -> cycle,
    "amber" -> amber,
    "offset" -> offset,
    "phases" -> phases,
    "greenTimes" -> greenTimes,
    "starts" -> starts
  |>
];

SignalIsGreen[controller_Association, edge_, simTime_] := Module[
  {t, phases, starts, greens, i},
  t = Mod[simTime - controller["offset"], controller["cycle"]];
  phases = controller["phases"];
  starts = controller["starts"];
  greens = controller["greenTimes"];
  Do[
    If[starts[[i]] <= t < starts[[i]] + greens[[i]] &&
       MemberQ[phases[[i]], edge],
      Return[True, Module]
    ],
    {i, Length[phases]}
  ];
  False
];

BuildSignalControllers[incomingEdgesAt_Association, nodeClass_Association,
                       cycle_: 90.0, seed_: 0] := Module[
  {controllers = <||>, node, cls, inEdges, offset},
  SeedRandom[seed];
  KeyValueMap[
    Function[{node, cls},
      If[cls === "signal",
        inEdges = Lookup[incomingEdgesAt, node, {}];
        If[inEdges =!= {},
          offset = RandomReal[{0, cycle}];
          controllers[node] =
            MakeSignalController[node, inEdges, cycle, 4.0, offset]
        ]
      ]
    ],
    nodeClass
  ];
  controllers
];

(* -------- roundabout gap acceptance -------- *)

(* circulatingOcc :: edge -> {{distToNode, leaderIndex}, ...}; leaderSpeeds :: leaderIndex -> speed;
   edgeLengths :: edge -> length. Returns True if ok to enter. *)
RoundaboutCanEnter[approachEdge_, circulatingOcc_Association,
                   leaderSpeeds_, edgeLengths_, criticalHeadway_: 2.5] := Module[
  {canEnter = True, occ, d, leader, spd, e, j},
  KeyValueMap[
    Function[{e, occ},
      If[e === approachEdge, Null,
        Do[
          leader = occ[[j, 2]]; d = occ[[j, 1]];
          spd = leaderSpeeds[leader];
          If[!NumberQ[spd] || spd < 0.5, Continue[]];
          If[d/Max[spd, 0.1] < criticalHeadway,
            canEnter = False; Break[]
          ],
          {j, Length[occ]}
        ]
      ]
    ],
    circulatingOcc
  ];
  canEnter
];

(* -------- virtual-gap delay for uncontrolled junctions -------- *)

IntersectionVirtualGap[nodeDegree_, nodeClass_String] := Module[{base, factor},
  base = Switch[nodeClass,
    "uncontrolled", 2.0,
    "giveWay",      4.0,
    "stop",         8.0,
    _,              2.0
  ];
  factor = 1.0 + 0.15 Max[nodeDegree - 3, 0];
  Max[base factor, 3.0]
];

(* -------- small demo: build a 4-approach controller and plot its schedule -------- *)

Options[SimulateSignalDemo] = {"cycle" -> 90.0, "amber" -> 4.0, "offset" -> 0.0,
  "nApproaches" -> 4};

SimulateSignalDemo[OptionsPattern[]] := Module[
  {nApp, cycle, amber, offset, approaches, ctrl, schedule, tAxis},
  nApp = OptionValue["nApproaches"]; cycle = OptionValue["cycle"];
  amber = OptionValue["amber"]; offset = OptionValue["offset"];
  approaches = Table[Symbol["approach" <> ToString[i]], {i, nApp}];
  ctrl = MakeSignalController["demoNode", approaches, cycle, amber, offset];
  tAxis = N @ Subdivide[0, 2 cycle, 720];
  schedule = Table[
    {tAxis, Table[
      Boole[SignalIsGreen[ctrl, approaches[[i]], t]],
      {t, tAxis}
    ]},
    {i, nApp}
  ];
  <|
    "controller" -> ctrl,
    "approaches" -> approaches,
    "schedule" -> schedule,
    "tAxis" -> tAxis,
    "cycle" -> cycle,
    "amber" -> amber
  |>
];

PlotSignalDemo[res_Association] := Module[
  {ctrl, approaches, schedule, cycle, amber, nApp, gantt, starts, greens,
   i, rows, colours, cycleLines},

  ctrl = res["controller"];
  approaches = res["approaches"];
  schedule = res["schedule"];
  cycle = res["cycle"];
  amber = res["amber"];
  nApp = Length[approaches];
  starts = ctrl["starts"];
  greens = ctrl["greenTimes"];
  colours = Table[ColorData["Rainbow"][(i - 1)/Max[nApp - 1, 1]], {i, nApp}];

  (* Gantt-style phase chart over 2 cycles *)
  rows = Flatten @ Table[
    Module[{s = starts[[i]], g = greens[[i]]},
      {
        (* green bar for cycle 1 and 2 *)
        {colours[[i]], EdgeForm[{Thin, Black}],
          Rectangle[{s, nApp - i + 0.1}, {s + g, nApp - i + 0.9}]},
        {Gray, EdgeForm[{Thin, Black}],
          Rectangle[{s + g, nApp - i + 0.1}, {s + g + amber, nApp - i + 0.9}]},
        {colours[[i]], EdgeForm[{Thin, Black}],
          Rectangle[{s + cycle, nApp - i + 0.1}, {s + g + cycle, nApp - i + 0.9}]},
        {Gray, EdgeForm[{Thin, Black}],
          Rectangle[{s + g + cycle, nApp - i + 0.1}, {s + g + amber + cycle, nApp - i + 0.9}]}
      }
    ],
    {i, nApp}
  ];
  cycleLines = {Dashed, Red, Line[{{cycle, 0}, {cycle, nApp}}]};

  gantt = Graphics[
    {rows, cycleLines,
      Table[Text[Style[Row[{"approach ", i}], 9],
        {-2, nApp - i + 0.5}, {1, 0}], {i, nApp}]
    },
    PlotRange -> {{-10, 2 cycle + 2}, {0, nApp + 0.5}},
    Axes -> {True, False},
    AxesOrigin -> {0, 0},
    AxesLabel -> {"time (s)", None},
    PlotLabel -> Row[{"Fixed-time signal schedule (cycle ", cycle,
                       "s, amber ", amber, "s, ", nApp, " approaches)"}],
    ImageSize -> 900,
    AspectRatio -> 0.35
  ];

  gantt
];

PlotSignalDemo[res_Association, path_String] := Module[{g = PlotSignalDemo[res]},
  Export[path, g, ImageResolution -> 150]; path];

End[];
EndPackage[];
