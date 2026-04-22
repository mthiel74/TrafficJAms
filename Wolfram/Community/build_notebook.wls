#!/usr/bin/env wolframscript
(* Build Wolfram/Community/trafficjams.nb -- a Wolfram Notebook suitable
   for posting to the Wolfram Community. Math is typeset natively via
   FormBox[..., TraditionalForm] (real fractions, Greek, partials, ...),
   and exported simultaneously as PDF for offline preview. *)

hereDir = DirectoryName[If[$ScriptCommandLine =!= {}, $InputFileName,
                           NotebookFileName[]]];
imgDir  = FileNameJoin[{hereDir, "..", "results"}];

imgWidth = 520;

(* For PNGs: embed the image directly. For GIFs: embed frames inside a
   ListAnimate so the notebook plays the animation with play / pause /
   scrubber controls. Frames are subsampled and resized to keep the .nb
   file under the 25 MB Community-post soft limit; the original
   full-resolution MP4 and GIF remain in Wolfram/results/. Individual
   animations can override the frame cap via $animFramesOverride. *)
maxFrames = 30;
animWidth = 420;

(* Animation-specific overrides keyed by filename. *)
$animFramesOverride = <|
  "anim_bando.gif" -> 80,
  "anim_nasch.gif" -> 40
|>;

(* Playback rate overrides (fps). Default is 4. Slower rates give the
   viewer more time to track what is happening in fast-evolving
   animations. *)
$animRateOverride = <|
  "anim_nasch.gif" -> 2,
  "anim_nasch_2lane.gif" -> 3
|>;

loadImg[name_] := With[
  {p = FileNameJoin[{imgDir, name}]},
  If[!FileExistsQ[p],
    Style["[missing image: " <> name <> "]", Red],
    If[StringEndsQ[ToLowerCase[name], ".gif"],
      With[{allFrames = Import[p, {"GIF", "ImageList"}],
            cap = Lookup[$animFramesOverride, name, maxFrames],
            rate = Lookup[$animRateOverride, name, 4]},
        With[{stride = Max[1, Ceiling[Length[allFrames]/cap]]},
          With[{frames = allFrames[[1 ;; -1 ;; stride]]},
            ListAnimate[
              ImageResize[#, {animWidth}] & /@ frames,
              AnimationRate -> rate,
              AnimationRunning -> True,
              AnimationRepetitions -> Infinity
            ]
          ]
        ]
      ],
      Image[Import[p], ImageSize -> imgWidth]
    ]
  ]
];

imgCell[name_, caption_String : ""] := Cell[
  BoxData[ToBoxes[
    If[caption === "",
      Pane[loadImg[name], imgWidth, Alignment -> Center],
      Column[{
        loadImg[name],
        Pane[Style[caption, Italic, FontSize -> 10,
                   FontFamily -> "Helvetica"],
             imgWidth, Alignment -> Center]
      }, Alignment -> Center, Spacings -> 0.6]
  ]]],
  "Output", ShowCellLabel -> False,
  TextAlignment -> Center,
  CellMargins -> {{20, 20}, {10, 14}}
];

(* ---------- math typesetting helpers ---------- *)

boldV = StyleBox["v", FontWeight -> Bold];

mathI[box_] := Cell[BoxData[FormBox[box, TraditionalForm]],
  "InlineFormula", FontFamily -> "Times"];

mathD[box_] := Cell[BoxData[FormBox[box, TraditionalForm]],
  "DisplayFormula", TextAlignment -> Center,
  FontFamily -> "Times"];

(* ---------- reusable math fragments ---------- *)

rhoXT       = RowBox[{"\[Rho]", "(", "x", ",", " ", "t", ")"}];
QofRho      = RowBox[{"Q", "(", "\[Rho]", ")"}];
rhoMax      = SubscriptBox["\[Rho]", "max"];
vMaxSym     = SubscriptBox["v", "max"];
rhoCrit     = SubscriptBox["\[Rho]", "c"];
vAlpha      = SubscriptBox["v", "\[Alpha]"];
sAlpha      = SubscriptBox["s", "\[Alpha]"];
dvAlpha     = RowBox[{"\[CapitalDelta]", SubscriptBox["v", "\[Alpha]"]}];
sStar       = SuperscriptBox["s", "*"];
s0Sym       = SubscriptBox["s", "0"];
THead       = "T";
v0Sym       = SubscriptBox["v", "0"];
Vopt        = SubscriptBox["V", "opt"];
kappa       = "\[Kappa]";
vMaxOVM     = SubscriptBox["v", "max"];
scSym       = SubscriptBox["s", "c"];
vMaxCell    = SubscriptBox["v", "max"];
pSlow       = "p";
lambda      = "\[Lambda]";
mu          = "\[Mu]";
rhoSym      = "\[Rho]";
LqSym       = SubscriptBox["L", "q"];
WqSym       = SubscriptBox["W", "q"];
te0         = SubsuperscriptBox["t", "e", "0"];
teX         = RowBox[{SubscriptBox["t", "e"], "(", SubscriptBox["x", "e"], ")"}];
xeSym       = SubscriptBox["x", "e"];
CeSym       = SubscriptBox["C", "e"];
alphaSym    = "\[Alpha]";
betaSym     = "\[Beta]";
tauSym      = "\[Tau]";
VeSym       = RowBox[{SubscriptBox["V", "e"], "(", "\[Rho]", ")"}];
c0Sym       = SubscriptBox["c", "0"];
Cycle       = "C";
greenG      = "g";
webRed      = RowBox[{"(", Cycle, "-", greenG, ")"}];

(* ---------- display equations ---------- *)

(* LWR conservation law *)
lwrEq = RowBox[{
  FractionBox[RowBox[{"\[PartialD]", "\[Rho]"}], RowBox[{"\[PartialD]", "t"}]],
  "+",
  FractionBox[RowBox[{"\[PartialD]", QofRho}], RowBox[{"\[PartialD]", "x"}]],
  "=", "0"}];

(* Greenshields fundamental diagram *)
greenshieldsEq = RowBox[{
  QofRho, "=", RowBox[{"\[Rho]", "\[ThinSpace]", vMaxSym,
    RowBox[{"(", RowBox[{"1", "-",
      FractionBox["\[Rho]", rhoMax]}], ")"}]}]}];

(* Godunov flux *)
godunovEq = RowBox[{"Q", SubsuperscriptBox["",
  RowBox[{"i", "+", RowBox[{"1", "/", "2"}]}], "G"], "=",
  RowBox[{
    RowBox[{"if", " ", SubscriptBox["\[Rho]", "L"], "\[LessEqual]",
      SubscriptBox["\[Rho]", "R"]}], ":",
    RowBox[{"min", "(",
      RowBox[{"Q", "(", SubscriptBox["\[Rho]", "L"], ")"}], ",",
      RowBox[{"Q", "(", SubscriptBox["\[Rho]", "R"], ")"}],
      ",", " ", "Q", "(", rhoCrit, ")", ")"}]}]}];

(* Payne-Whitham second-order momentum equation *)
pwLHS = RowBox[{
  FractionBox[RowBox[{"\[PartialD]", "v"}], RowBox[{"\[PartialD]", "t"}]],
  "+", "v",
  FractionBox[RowBox[{"\[PartialD]", "v"}], RowBox[{"\[PartialD]", "x"}]]
}];

payneEq = RowBox[{
  pwLHS, "=",
  FractionBox[RowBox[{VeSym, "-", "v"}], tauSym],
  "-",
  FractionBox[SuperscriptBox[c0Sym, "2"], "\[Rho]"],
  FractionBox[RowBox[{"\[PartialD]", "\[Rho]"}], RowBox[{"\[PartialD]", "x"}]]
}];

(* IDM acceleration *)
idmAccEq = RowBox[{
  OverscriptBox[vAlpha, "\[CenterDot]"], "=",
  RowBox[{"a", " ",
    RowBox[{"[",
      RowBox[{"1", "-",
        SuperscriptBox[RowBox[{"(", FractionBox[vAlpha, v0Sym], ")"}], "\[Delta]"],
        "-",
        SuperscriptBox[RowBox[{"(",
          FractionBox[RowBox[{sStar,
            "(", vAlpha, ",", " ", dvAlpha, ")"}], sAlpha], ")"}], "2"]}],
      "]"}]}]}];

(* IDM desired-gap *)
idmGapEq = RowBox[{
  RowBox[{sStar, "(", "v", ",", " ", "\[CapitalDelta]v", ")"}],
  "=",
  RowBox[{s0Sym, "+", RowBox[{"v", "\[ThinSpace]", THead}],
    "+",
    FractionBox[RowBox[{"v", "\[ThinSpace]", "\[CapitalDelta]v"}],
      RowBox[{"2", SqrtBox[RowBox[{"a", "\[ThinSpace]", "b"}]]}]]}]}];

(* Bando OVM *)
bandoEq = RowBox[{
  OverscriptBox[vAlpha, "\[CenterDot]"], "=",
  RowBox[{kappa, " ",
    RowBox[{"[", RowBox[{Vopt, "(", sAlpha, ")", "-", vAlpha}], "]"}]}]}];

bvInner = RowBox[{
  "(",
  RowBox[{"tanh", "(",
    RowBox[{FractionBox["s", scSym], "-", "2"}], ")"}],
  "+",
  RowBox[{"tanh", "(", "2", ")"}],
  ")"
}];

bvNumerator = RowBox[{vMaxOVM, "\[ThinSpace]", bvInner}];

bvDenominator = RowBox[{"1", "+", RowBox[{"tanh", "(", "2", ")"}]}];

bandoVopt = RowBox[{
  RowBox[{Vopt, "(", "s", ")"}],
  "=",
  FractionBox[bvNumerator, bvDenominator]
}];

(* NaSch CA rules (itemised as a math display) *)
naschRule1 = RowBox[{"(", "1", ")", " ", "accelerate:", " ",
  "v", "\[Rule]", RowBox[{"min", "(", RowBox[{"v", "+", "1"}], ",", " ", vMaxCell, ")"}]}];
naschRule2 = RowBox[{"(", "2", ")", " ", "brake:", " ",
  "v", "\[Rule]", RowBox[{"min", "(", "v", ",", " ",
    RowBox[{"d", "-", "1"}], ")"}]}];
naschRule3 = RowBox[{"(", "3", ")", " ", "randomise:", " ",
  RowBox[{"with", " ", "probability", " ", pSlow}], ",", " ",
  "v", "\[Rule]", RowBox[{"max", "(", RowBox[{"v", "-", "1"}], ",", " ", "0", ")"}]}];
naschRule4 = RowBox[{"(", "4", ")", " ", "move:", " ",
  "x", "\[Rule]", RowBox[{"x", "+", "v"}]}];

(* M/D/1 queueing *)
md1LqEq = RowBox[{LqSym, "=",
  FractionBox[SuperscriptBox[rhoSym, "2"],
    RowBox[{"2", "(", RowBox[{"1", "-", rhoSym}], ")"}]]}];

md1WqEq = RowBox[{WqSym, "=",
  FractionBox[rhoSym,
    RowBox[{"2", "\[ThinSpace]", mu, "\[ThinSpace]",
      RowBox[{"(", RowBox[{"1", "-", rhoSym}], ")"}]}]]}];

rhoDef = RowBox[{rhoSym, "=", FractionBox[lambda, mu]}];

(* BPR cost *)
bprEq = RowBox[{teX, "=",
  RowBox[{te0, " ",
    RowBox[{"[",
      RowBox[{"1", "+",
        RowBox[{alphaSym, " ",
          SuperscriptBox[RowBox[{"(", FractionBox[xeSym, CeSym], ")"}],
            betaSym]}]}],
      "]"}]}]}];

(* Beckmann objective *)
beckmannEq = RowBox[{
  RowBox[{UnderscriptBox["min", "x"]}], " ",
  RowBox[{
    UnderoverscriptBox["\[Sum]", RowBox[{"e", "\[Element]", "E"}], ""], " ",
    SubsuperscriptBox["\[Integral]", "0", xeSym], " ",
    RowBox[{SubscriptBox["t", "e"], "(", "\[Omega]", ")"}],
    RowBox[{"\[DifferentialD]", "\[Omega]"}]}]}];

wardropEq = RowBox[{SubscriptBox["c", "p"], "=", SubscriptBox["c", "q"],
  "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  RowBox[{"\[ForAll]", " ", "used", " ", "paths", " ", "p", ",", " ", "q"}]}];

(* Webster uniform delay *)
websterEq = RowBox[{"d", SubscriptBox["", "unif"], "=",
  FractionBox[SuperscriptBox[webRed, "2"], RowBox[{"2", Cycle}]]}];

(* cells *)

cells = {

 Cell["Traffic Flow Dynamics in the Wolfram Language",
      "Title"],

 Cell[TextData[{"From phantom jams on a ring road to a full-city \
multi-agent simulation of Aberdeen \[LongDash] a self-contained \
port of classical traffic-flow models."}],
      "Subtitle"],

 Cell[TextData[{"Marco Thiel, April 2026 \[LongDash] ",
    ButtonBox["github.com/mthiel74/TrafficJAms",
      BaseStyle -> "Hyperlink",
      ButtonData -> {URL["https://github.com/mthiel74/TrafficJAms"],
                     None}]}],
      "Text", TextAlignment -> Center, FontSlant -> Italic],

 (* ---- Abstract ---- *)

 Cell["Abstract", "Section"],

 Cell[TextData[{
   "Traffic jams are the canonical everyday example of an emergent \
phenomenon: each driver follows simple local rules, and the system \
as a whole spontaneously produces stop-and-go waves, phantom jams, \
shockwaves, and city-scale gridlock. Since the 1950s these have been \
modelled at wildly different scales \[LongDash] as a hyperbolic PDE \
(Lighthill\[Dash]Whitham\[Dash]Richards), as individual vehicles \
following an ODE (Intelligent Driver Model, Bando OVM), as a \
stochastic cellular automaton (Nagel\[Dash]Schreckenberg), as a \
queueing network at intersections, or as a convex optimisation \
problem on a directed graph (Wardrop equilibrium). This notebook \
walks through all of those in turn, each with a short derivation, a \
Wolfram-Language implementation, and output from a real simulation. \
The running case study is Aberdeen in north-east Scotland \
\[LongDash] a city whose peculiar geography (squeezed between the \
Rivers Don and Dee, the coast, and the hills, with two critical \
bridges) makes it an unusually nice traffic-engineering laboratory. \
The final section applies the full pipeline to the 11\\[ThinSpace]925\
-node Aberdeen drivable road network extracted from OpenStreetMap, \
with 600 agents, UK-standard 90\[ThinSpace]s signal cycles, and \
roundabout gap acceptance."}], "Text"],

 imgCell["aberdeen_full.png",
   "Figure 1. 600 vehicles on the full Aberdeen road network \
(11,925 drivable nodes, 26,453 directed edges after strongly-\
connected-component restriction). Vehicles follow the Intelligent \
Driver Model with hierarchy-weighted noisy routing and staggered \
entry; 151 OSM-tagged signalised junctions and ~1000 \
roundabout-tagged edges enforce intersection control. Colours \
encode instantaneous speed (green = free flow, red = stopped)."],

 (* ---- Section 1 ---- *)

 Cell["1. Why model traffic?", "Section"],

 Cell[TextData[{
   "Traffic engineers need quantitative tools to answer questions \
like: What happens to queue lengths on Union Street if we add a bus \
gate? If the Bridge of Don closes for maintenance, does Anderson \
Drive absorb the flow, or does half the city seize up? What is the \
shortest green phase for the busy intersection at Mounthooly that \
still keeps average delay under 30\[ThinSpace]s? None of these \
questions has a closed-form answer, but all of them are tractable \
in simulation once the correct mathematical model has been picked \
for the right spatial scale."}], "Text"],

 Cell[TextData[{
   "There is no single \"right\" model of traffic; there is a \
hierarchy of models, each trading detail for tractability. At the \
coarsest end we treat traffic as a compressible fluid; at the \
finest end we simulate every vehicle with its own ODE. In between \
sit stochastic cellular automata and queueing approximations. This \
notebook introduces each of these scales, shows the canonical \
model for it, and ends by stitching them all together on the real \
Aberdeen street map."}], "Text"],

 Cell[TextData[{"Aberdeen is a good running example for three \
reasons: it is small enough to be tractable (pop. ~200\[ThinSpace]\
000, compact historic centre), large enough to be interesting \
(international airport, oil-industry commuter flows, a peripheral \
ring road), and its geography creates natural bottlenecks \
\[LongDash] the two main river bridges, the Haudagain roundabout, \
the Anderson Drive corridor \[LongDash] that any model will want \
to reproduce."}], "Text"],

 (* ---- Section 2: LWR ---- *)

 Cell["2. The macroscopic view: the LWR model", "Section"],

 Cell[TextData[{"Lighthill and Whitham (1955), followed independently \
by Richards (1956), proposed that traffic on a long road obeys the \
same conservation law as any other compressible one-dimensional \
fluid: vehicles cannot be created or destroyed, so the density ",
   mathI[rhoXT],
   " and flow ",
   mathI[QofRho],
   " satisfy"}], "Text"],

 mathD[lwrEq],

 Cell[TextData[{"This is a first-order hyperbolic PDE. The missing \
ingredient is the ",
   StyleBox["fundamental diagram", FontSlant -> Italic],
   " \[LongDash] an empirical relation between density and flow. \
The simplest, due to Greenshields (1935), is linear in the speed:"}],
      "Text"],

 mathD[greenshieldsEq],

 Cell[TextData[{"so that flow is zero both at empty road (\[Rho]=0) \
and at bumper-to-bumper jam (\[Rho]=",
   mathI[rhoMax],
   "), and is maximised at the critical density ",
   mathI[RowBox[{rhoCrit, "=", RowBox[{rhoMax, "/", "2"}]}]],
   ". Plugging this into the conservation law gives a nonlinear \
scalar PDE that develops shocks from smooth initial data: the \
mathematical origin of a traffic jam."}], "Text"],

 Cell[TextData[{"We discretise it with the Godunov finite-volume \
scheme (the simplest upwind flux that respects the shock \
structure). The numerical flux between cells with left density ",
   mathI[SubscriptBox["\[Rho]", "L"]], " and right density ",
   mathI[SubscriptBox["\[Rho]", "R"]], " is"}], "Text"],

 mathD[godunovEq],

 Cell[TextData[{"with the usual case analysis for \[Rho]\
\\[ThinSpace]L > \[Rho]\\[ThinSpace]R. A CFL condition \
(\[CapitalDelta]t \[LessEqual] \[CapitalDelta]x / ",
   mathI[vMaxSym],
   ") keeps the scheme stable. The Wolfram implementation is a \
dozen lines:"}], "Text"],

 Cell[BoxData[
"Greenshields[rho_, rhoMax_, vMax_] := rho vMax (1 - rho/rhoMax);\n\
GodunovFlux[rhoL_, rhoR_, rhoMax_, vMax_] := Module[{rhoCrit = rhoMax/2,\n\
    qL = Greenshields[rhoL, rhoMax, vMax], qR = Greenshields[rhoR, rhoMax, vMax]},\n\
  Which[\n\
    rhoL <= rhoR,\n\
      Which[rhoL >= rhoCrit, qL, rhoR <= rhoCrit, qR,\n\
            True, Greenshields[rhoCrit, rhoMax, vMax]],\n\
    True,\n\
      If[qL <= qR, qL, If[rhoL >= rhoCrit, qL, Max[qL, qR]]]]];"],
      "Input"],

 Cell[TextData[{"Seeding an initial high-density block ",
   mathI[rhoSym],
   "=120 veh/km in the middle of an otherwise uniform \
road at ",
   mathI[rhoSym],
   "=20 veh/km produces the canonical signature of an LWR jam: a \
backward-travelling shock (upstream boundary of the jam) and a \
downstream rarefaction fan (vehicles leaving the jam into free \
road)."}], "Text"],

 imgCell["anim_lwr.gif",
   "Figure 2. LWR density evolution on a 10 km corridor with an \
initial high-density block. The hot-coloured jam block disperses \
into a backward-propagating shockfront (visible travelling \
leftwards) and a downstream rarefaction fan; the backward \
propagation speed equals the slope of the shock line in the \
underlying space-time diagram. (The MP4 version is at \
Wolfram/results/anim_lwr.mp4.)"],

 (* ---- Section 3: Payne-Whitham ---- *)

 Cell["3. Adding a second equation: Payne-Whitham", "Section"],

 Cell[TextData[{"A weakness of the LWR model is that velocity is \
always the instantaneous equilibrium value on the fundamental \
diagram \[LongDash] drivers adapt infinitely fast. Real drivers \
take a second or two to react, and they anticipate the state of \
traffic ahead. Payne (1971) and Whitham (1974) independently \
proposed a second equation \[LongDash] a momentum equation \
\[LongDash] in which velocity relaxes with time constant ",
   mathI[tauSym],
   " towards its equilibrium ",
   mathI[VeSym],
   " and is pushed by a pressure-like term ",
   mathI[SuperscriptBox[c0Sym, "2"]],
   " representing anticipation:"}], "Text"],

 mathD[payneEq],

 Cell[TextData[{"coupled to the same continuity equation as LWR. \
The system is now second-order and supports stop-and-go waves: \
velocity is no longer slaved to density. In our numerical \
experiment we seed a congested block on an otherwise \
moderate-density corridor \[LongDash] the kind of configuration \
you get at an on-ramp merge \[LongDash] and watch the oscillatory \
structure develop."}], "Text"],

 imgCell["anim_payne_whitham.gif",
   "Figure 3. Animated Payne-Whitham run near a merge zone. \
Unlike LWR the velocity field can drop below equilibrium \
transiently, producing stop-and-go oscillations rather than a \
clean shock. (MP4: Wolfram/results/anim_payne_whitham.mp4.)"],

 (* ---- Section 4: IDM ---- *)

 Cell["4. The microscopic view: Intelligent Driver Model",
      "Section"],

 Cell[TextData[{"Zooming in to the level of an individual vehicle \
\[LongDash] indexed ",
   mathI["\[Alpha]"],
   " \[LongDash] the Intelligent Driver Model (Treiber, Hennecke & \
Helbing, 2000) is a second-order ODE for the longitudinal \
dynamics. It is written as"}], "Text"],

 mathD[idmAccEq],

 Cell[TextData[{"where ",
   mathI[vAlpha], " and ", mathI[sAlpha],
   " are the speed and gap to the vehicle in front, ",
   mathI[dvAlpha],
   " is the approach rate, and the desired gap is"}], "Text"],

 mathD[idmGapEq],

 Cell[TextData[{"The constants have direct physical meaning: ",
   mathI[v0Sym], " is the desired free-flow speed (typically \
15\[ThinSpace]m/s on an urban road), ",
   mathI[s0Sym], " is the minimum bumper gap at standstill (2\
\[ThinSpace]m), ",
   mathI[THead],
   " is the safe time headway to the leader (1.5\[ThinSpace]s), and \
",
   mathI["a"], ", ", mathI["b"],
   " are the maximum acceleration and comfortable braking \
(1 and 1.5 m/s", mathI[SuperscriptBox["", "2"]],
   "). The acceleration exponent ", mathI["\[Delta]"],
   " (typically 4) controls how sharply a driver eases off the \
accelerator as they approach their desired speed."}], "Text"],

 Cell[TextData[{"Running a strongly unstable configuration \
\[LongDash] 25 vehicles on a 1\[ThinSpace]km ring (40 m mean gap), \
desired speed ",
   mathI[RowBox[{v0Sym, "=30"}]], " m/s, time headway ",
   mathI[RowBox[{"T", "=2.5"}]], " s (so each driver wants ~75 m of \
gap), sluggish acceleration ",
   mathI[RowBox[{"a", "=0.3"}]], " m/",
   mathI[SuperscriptBox["s", "2"]], ", and firm braking ",
   mathI[RowBox[{"b", "=2.0"}]], " m/",
   mathI[SuperscriptBox["s", "2"]], " \[LongDash] reproduces one of \
the most pedagogically useful results in all of traffic science: \
the ",
   StyleBox["phantom jam", FontSlant -> Italic],
   ". A single-vehicle speed perturbation is not absorbed; instead \
it amplifies through the platoon into a persistent \
backward-propagating jam. The animation shows this both on the \
ring (deep-red cluster of near-stopped cars, sparse green free-flow \
arc) and in the accumulating space-time diagram below it, where \
the jam waves are the diagonal red/orange bands."}],
      "Text"],

 Cell[TextData[{
   "This is exactly the same phenomenon that Sugiyama et al. \
(2008) demonstrated experimentally on a real circular track \
\[LongDash] their striking video is worth watching alongside the \
simulation: ",
   ButtonBox["youtube.com/watch?v=Rryu85BtALM",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://www.youtube.com/watch?v=Rryu85BtALM"],
                    None}],
   "."}],
      "Text"],

 imgCell["anim_idm.gif",
   "Figure 4. 25 IDM vehicles on a 1 km ring with an extreme \
phantom-jam configuration (v0=30 m/s, T=2.5 s, a=0.3 m/s^2, \
b=2.0 m/s^2). At t=0 the cars are visibly spaced out (40 m mean \
gap) at free-flow speed. A single-vehicle speed perturbation \
amplifies; by t=400 s the ring has a deep-red cluster of \
near-stopped cars (slowest ~0.8 m/s) next to a green free-flow \
arc (fastest ~17 m/s). Bottom panel: accumulating space-time \
trajectory diagram (time along x, position along y with \
wrap-around at 1000 m). The backward-travelling jam wave is the \
diagonal red band - cars move forward (slope up-right) but the \
jam drifts backward relative to them. (MP4: \
Wolfram/results/anim_idm.mp4.)"],

 (* ---- Section 5: Bando OVM ---- *)

 Cell["5. The simplest car-following that jams: Bando OVM",
      "Section"],

 Cell[TextData[{"Even simpler than IDM, the Optimal Velocity Model \
(Bando et al. 1995) relaxes speed towards a target velocity that \
depends only on the gap to the vehicle ahead:"}], "Text"],

 mathD[bandoEq],

 Cell[TextData[{"with the standard choice"}], "Text"],

 mathD[bandoVopt],

 Cell[TextData[{"The model has a single stability parameter, the \
sensitivity ",
   mathI[kappa],
   ". For ",
   mathI[kappa],
   " above a critical value the uniform-flow solution is stable; \
below it, uniform flow is linearly unstable via a Hopf bifurcation \
and the system collapses into stop-and-go waves. This is the \
cleanest mathematical explanation for why phantom jams exist: they \
are the attractor of a dynamical bifurcation."}], "Text"],

 imgCell["anim_bando.gif",
   "Figure 5. Bando OVM on an 800 m ring with 18 vehicles and \
kappa=0.4, below the stability threshold 2 V_opt'(s*) ~ 0.61. \
Top: ring snapshot showing the orange-red stop-and-go cluster and \
green free-flow regions. Bottom: accumulating space-time trajectory \
diagram \[Dash] the diagonal orange bands drifting backward \
relative to the car trajectories are the stop-and-go waves. Speed \
spread grows over time as the Hopf instability saturates. (MP4: \
Wolfram/results/anim_bando.mp4.)"],

 (* ---- Section 6: NaSch ---- *)

 Cell["6. A stochastic cellular automaton: Nagel-Schreckenberg",
      "Section"],

 Cell[TextData[{"Nagel and Schreckenberg (1992) gave the minimal CA \
that reproduces realistic traffic dynamics. Space is discretised \
into 7.5\[ThinSpace]m cells, time into 1\[ThinSpace]s steps, and \
velocity is an integer in [0, ",
   mathI[vMaxCell],
   "]. At each step every vehicle applies four rules:"}], "Text"],

 mathD[naschRule1],
 mathD[naschRule2],
 mathD[naschRule3],
 mathD[naschRule4],

 Cell[TextData[{"where ", mathI["d"],
   " is the number of empty cells to the next vehicle. The \
stochastic slow-down rule (3) is essential: without it the model \
reduces to a deterministic CA and does not jam spontaneously. With \
it, the space-time diagram spontaneously generates \
backward-propagating \"wave\" bands, matching the LWR and IDM \
results from a completely different starting point."}], "Text"],

 imgCell["anim_nasch.gif",
   "Figure 6. 100 NaSch vehicles on a 500-cell periodic lattice \
with randomisation probability p=0.3 and vMax=5 cells/step. Top \
panel: current positions, coloured by speed. Bottom panel: the \
space-time diagram grows line by line, and backward-propagating \
jam waves appear as diagonal dark bands. (MP4: \
Wolfram/results/anim_nasch.mp4.)"],

 Cell[TextData[{"Extending to two lanes with a symmetric \
lane-change rule (incentive: would have to brake in current lane; \
safety: sufficient gap ahead and behind in the target lane; \
target cell free) recovers the classic multi-lane \
redistribution."}], "Text"],

 imgCell["anim_nasch_2lane.gif",
   "Figure 7. Two-lane NaSch with 150 vehicles (75 per lane \
initially). The running counter in the title shows cumulative \
lane-change events; each occurs when a vehicle would otherwise \
have to brake but the other lane satisfies the gap-acceptance \
criteria. (MP4: Wolfram/results/anim_nasch_2lane.mp4.)"],

 imgCell["nasch_2lane.png",
   "Figure 8. Static space-time view of the same 2-lane run. Left \
two panels: per-lane space-time diagrams. Right: number of lane \
changes per step. Peaks in lane-change activity coincide with \
density imbalances between the two lanes."],

 (* ---- Section 7: queues ---- *)

 Cell["7. Intersections as queues: M/D/1", "Section"],

 Cell[TextData[{"A signalised intersection looks like a server \
that alternates between available (green) and unavailable (red). \
If vehicle arrivals are Poisson with rate ",
   mathI[lambda],
   " and service during the green phase is deterministic at rate ",
   mathI[mu],
   ", the steady-state queue length is given by the \
Pollaczek\[Dash]Khinchine formula for an M/D/1 queue,"}], "Text"],

 mathD[md1LqEq],

 Cell[TextData[{"with the same-shape result for the waiting time:"}],
      "Text"],

 mathD[md1WqEq],

 Cell[TextData[{"both diverging as utilisation ",
   mathI[rhoDef],
   " approaches 1. Traffic engineers read the practical threshold \
",
   mathI[RowBox[{rhoSym, "=", "0.85"}]],
   " as the onset of severe congestion. On a tandem of ",
   mathI["n"],
   " signalised intersections along a corridor we approximate \
total delay by summing the per-intersection delays (a Jackson-\
network approximation that ignores platoon-preservation between \
signals but is accurate enough for first-pass planning)."}],
      "Text"],

 imgCell["queueing.png",
   "Figure 9. M/D/1 queueing on a four-intersection corridor. \
Top: per-intersection queue length (left) and waiting time (right) \
as a function of utilisation; both blow up as rho -> 1. Bottom: \
total queue and total delay for a 4-intersection tandem. The red \
dashed line marks the rho=0.85 severe-congestion onset."],

 (* ---- Section 8: signal opt ---- *)

 Cell["8. Optimising the signal timing", "Section"],

 Cell[TextData[{"Given a corridor of signals at which arrivals \
vary by intersection, what green times minimise total delay, \
subject to a shared green-time budget? Delay at each intersection \
combines the M/D/1 waiting term above with Webster's uniform \
delay for the fraction of the cycle spent red,"}], "Text"],

 mathD[websterEq],

 Cell[TextData[{"where ", mathI[Cycle], " is cycle length and ",
   mathI[greenG],
   " is green time at the intersection. Minimising the sum of \
these over green times subject to a total budget is a smooth \
convex problem and ",
   StyleBox["FindMinimum[..., Method -> \"InteriorPoint\"]", "Input"],
   " solves it in milliseconds. On a five-intersection Union \
Street-like corridor with arrival rates (0.08, 0.15, 0.22, 0.17, \
0.06), the optimum re-allocates green from the two quiet \
intersections to the central busy one and cuts total delay by \
~8.4 %."}], "Text"],

 imgCell["signal_optimisation.png",
   "Figure 10. Webster-style green-time optimisation. Left: \
uniform (50 s each) versus optimised green phase allocation \
across the five intersections. Middle: total delay drops from \
8.44 to 7.73 veh s (8.4% reduction). Right: sensitivity at the \
busiest intersection showing the U-shape that gives rise to the \
optimum."],

 imgCell["signal_schedule.png",
   "Figure 11. Gantt chart of a 90 s cycle / 4 s amber / 4-approach \
round-robin schedule. Each approach gets 18.5 s of green per cycle, \
separated by amber intergreens. The controller is the building \
block for signal control in the Aberdeen multi-agent simulation \
further down."],

 (* ---- Section 9: network assignment ---- *)

 Cell["9. Network-level assignment: Wardrop equilibrium", "Section"],

 Cell[TextData[{"Zooming back out: at the scale of a whole city, \
the question is no longer \"what does one link look like?\" but \
\"how do drivers distribute themselves across the network?\". The \
standard answer is Wardrop's first principle (1952): at \
equilibrium, all used paths between the same origin-destination \
pair have equal cost; no driver can unilaterally improve by \
switching routes. Formally, for all paths ",
   mathI["p, q"],
   " used by the same OD pair,"}], "Text"],

 mathD[wardropEq],

 Cell[TextData[{"Beckmann (1956) showed that Wardrop equilibrium \
link flows are the unique minimiser of the convex program"}],
      "Text"],

 mathD[beckmannEq],

 Cell[TextData[{"where ",
   mathI[xeSym], " is the flow on link ", mathI["e"],
   " and ",
   mathI[RowBox[{SubscriptBox["t", "e"], "(\[CenterDot])"}]],
   " is the increasing link-cost function. The universal practical \
choice is the BPR formula of the US Bureau of Public Roads,"}],
      "Text"],

 mathD[bprEq],

 Cell[TextData[{"with ",
   mathI[te0],
   " the free-flow travel time, ",
   mathI[CeSym],
   " the capacity, and tuning parameters ",
   mathI[RowBox[{alphaSym, "=", "0.15"}]],
   ", ",
   mathI[RowBox[{betaSym, "=", "4"}]],
   ". The objective is smooth and strictly convex, so \
",
   StyleBox["FindMinimum", "Input"], " solves it in a handful of \
iterations."}], "Text"],

 Cell[TextData[{"On a hand-curated 15-node Aberdeen network \
(Haudagain, Bridge of Don, Bridge of Dee, Union Street / Market \
Street, Anderson Drive, AWPR junction, etc.) with four \
origin-destination pairs summing to 8\[ThinSpace]500 \
veh/h, the solver identifies Union Street and Anderson Drive \
(south) as the dominant bottlenecks \[LongDash] saturation values \
of 3.0\[Times] and 2.4\[Times] capacity respectively, matching \
the real-world experience of commuters."}], "Text"],

 imgCell["network_assignment.png",
   "Figure 12. Wardrop-equilibrium flow on a toy 8-node Aberdeen-\
inspired network. Left: network flow map; edge thickness and \
colour encode flow. Right: path-flow distribution; all used paths \
(1, 2, 4) end up with approximately equal costs ~19.6 while the \
unused path 3 has higher cost - exactly Wardrop's principle."],

 imgCell["aberdeen_network.png",
   "Figure 13. Equilibrium on the realistic 15-node Aberdeen \
network. Left: saturation map of the 23 roads; the thickest red \
edge is Union Street. Right: top ten most-congested links, with \
Union Street at 3.0x capacity followed by Anderson Drive (south) \
and Berryden Road."],

 (* ---- Section 10: dynamic assignment ---- *)

 Cell["10. Time-varying demand", "Section"],

 Cell[TextData[{"Real rush-hour demand is not a single number; it \
is a Gaussian-ish profile that builds up, peaks, and decays. We \
therefore run the Wardrop solve period by period, with some \
fraction of each period's flow carrying over as residual \
congestion that reduces the effective capacity for the next \
period:"}], "Text"],

 Cell[BoxData[
"effectiveCap[e, t+1] = Max[cap[e] - 0.3 residual[e, t], 0.5 cap[e]];\n\
(* then solve Beckmann on (fft, effectiveCap) with that period's demand *)"],
      "Input"],

 Cell[TextData[{"This is a very coarse approximation to full \
Dynamic Traffic Assignment but captures the right qualitative \
behaviour: flows on the shortest route pile up first, spill over \
to secondary routes as the fastest route saturates, and decay \
asymmetrically on the way out of the peak because residual \
congestion persists."}], "Text"],

 imgCell["dynamic_assignment.png",
   "Figure 14. 12-period dynamic assignment (period length 5 min) \
on the toy Aberdeen network. Top-left: Gaussian-shaped demand \
profile peaking mid-morning. Top-right: path-flow evolution. \
Bottom-left: flow on four key links. Bottom-right: link-cost \
heatmap showing congestion spread and persistence."],

 (* ---- Section 11: full Aberdeen ---- *)

 Cell["11. The full case study: Aberdeen from OpenStreetMap",
      "Section"],

 Cell[TextData[{"All the ingredients above stitch together to \
produce a realistic multi-agent simulation of the full Aberdeen \
drivable road network. The network is parsed from an Overpass API \
JSON extract (council-area-wide query) by filtering for drivable \
",
   StyleBox["highway=*", FontFamily -> "Courier"],
   " tags and splitting each way at intersection nodes, which \
leaves an 11\[ThinSpace]925-node, 26\[ThinSpace]453-edge directed \
graph after restricting to the largest strongly-connected \
component. 151 OSM-tagged traffic-signal junctions get a \
fixed-time controller with a 60\[ThinSpace]s cycle, 4\
\[ThinSpace]s amber, and a randomised offset. ~1000 edges tagged \
",
   StyleBox["junction=roundabout", FontFamily -> "Courier"],
   " enforce gap acceptance via a TRL\\[ThinSpace]281-style \
critical headway."}], "Text"],

 Cell[TextData[{"Every vehicle carries a full IDM car-following \
state (a=2, b=3, s0=2, T=1.2, ",
   mathI[RowBox[{"\[Delta]", "=", "4"}]],
   ", with the velocity-difference term). Origin-destination pairs \
are sampled by node degree so trips cluster at major intersections. \
Routes are computed on a graph whose edge weights are length \
times a per-highway-class factor (0.5 for motorway, 1.2 for \
residential, \[Ellipsis]) times a frozen per-edge random factor in \
[0.8, 1.5], giving genuine route diversity without rebuilding the \
graph on every call. Entries are staggered over the first \
60\[ThinSpace]s, and at each junction there is a 10% chance of a \
probabilistic detour. Within 15\[ThinSpace]m of the stop line a \
red signal or a blocked roundabout caps the vehicle's \
speed-limit-for-IDM at a distance-proportional value, causing the \
IDM to decelerate naturally towards the stop."}], "Text"],

 Cell[BoxData[
"Get[\"IntersectionControl.wl\"];\n\
Get[\"AberdeenFullCity.wl\"];\n\
net = AberdeenFullCity`LoadFullAberdeen[];\n\
sim = AberdeenFullCity`SimulateFullAberdeen[net,\n\
   \"nVehicles\" -> 600, \"nFrames\" -> 120, \"T\" -> 360];\n\
AberdeenFullCity`PlotFullAberdeen[sim, \"aberdeen_full.png\"];\n\
AberdeenFullCity`AnimateFullAberdeen[sim, \"aberdeen_full.gif\"];"],
      "Input"],

 Cell[TextData[{"On an M-class laptop this pipeline takes about \
30\[ThinSpace]s for the simulation and a minute for the GIF \
export. The final frame at ",
   mathI[RowBox[{"t", "=", "360 s"}]],
   " shows the Aberdeen Western Peripheral Route on the outer \
ring, heavy flows along the A90 down the east coast and into the \
city, and congestion at the expected pinch points."}], "Text"],

 imgCell["aberdeen_full.gif",
   "Figure 15. Animated full-city simulation (600 vehicles, \
12 fps, 36 frames). Signal-controlled junctions impose the \
expected wave-structure on the arterial roads. The AWPR ring is \
clearly visible on the west/north perimeter and carries the \
lion's share of through-flow, exactly as intended when it opened \
in 2018. (MP4: Wolfram/results/aberdeen_full.mp4.)"],

 imgCell["aberdeen_city.gif",
   "Figure 16. Zoomed-in view: 150 vehicles on the 800 m radius \
city-centre extract (1397 drivable nodes). Union Street cuts \
diagonally through the middle; Marischal College is at the \
centre. At this scale individual vehicle interactions and queue \
build-up at intersections are very visible. (MP4: \
Wolfram/results/aberdeen_city.mp4.)"],

 (* ---- Section 12: implementation notes ---- *)

 Cell["12. Implementation notes", "Section"],

 Cell[TextData[{"A few Wolfram-Language design choices that make \
the code short and fast:"}], "Text"],

 Cell[TextData[{
   "\[Bullet] ",
   "Each model lives in its own ",
   StyleBox[".wl", "Input"],
   " package under ",
   StyleBox["Wolfram/", "Input"],
   " with a ",
   StyleBox["Simulate<Name>[opts]", "Input"], " and ",
   StyleBox["Plot<Name>[result]", "Input"],
   " entry point. The top-level ",
   StyleBox["RunAll.wls", "Input"],
   " runs every model in ~40\[ThinSpace]s end to end.\n\[Bullet] ",
   "IDM, Bando, and LWR use vectorised ",
   StyleBox["RotateLeft", "Input"],
   " to get the leader\[Hyphen]follower gaps in one call instead \
of a Python-style loop.\n\[Bullet] ",
   "The NaSch lane-changing rule uses a ",
   StyleBox["Dispatch", "Input"],
   "-backed membership test for occupied target cells, which makes \
the inner loop practically constant-time.\n\[Bullet] ",
   "Both assignment models (",
   StyleBox["NetworkAssignment", "Input"], ", ",
   StyleBox["DynamicAssignment", "Input"],
   ") use ",
   StyleBox["FindMinimum[{obj, constraints}, vars, Method -> \
\"InteriorPoint\"]", "Input"],
   " for the Beckmann minimisation, exploiting that it is a \
smooth strictly convex problem.\n\[Bullet] ",
   "The full-city simulator builds a single hierarchy-weighted \
graph once with per-edge random multipliers in ",
   mathI[RowBox[{"[", "0.8", ",", "1.5", "]"}]],
   ", giving frozen but distinct routes per vehicle. Shortest \
paths then use Wolfram's ",
   StyleBox["FindShortestPath", "Input"],
   " on this one graph; the full 600-vehicle run never touches the \
Graph[] constructor in the inner loop.\n\[Bullet] ",
   "Signal phase logic is a tiny data-driven ",
   StyleBox["Association", "Input"],
   " keyed by node: ",
   StyleBox["SignalIsGreen[ctrl, edge, simTime]", "Input"],
   " is a handful of ",
   StyleBox["Mod", "Input"],
   " / ",
   StyleBox["MemberQ", "Input"],
   " calls, not a class hierarchy.\n\[Bullet] ",
   "Animations use ",
   StyleBox["Export[path, Table[frame, ...], \"DisplayDurations\" -> \
...]", "Input"],
   " \[LongDash] no external ImageMagick or ffmpeg call is \
needed."}], "Text"],

 (* ---- Section 13: references ---- *)

 Cell["13. References", "Section"],

 Cell[TextData[{
   "\[Bullet] ",
   "M. J. Lighthill & G. B. Whitham (1955), \"On kinematic waves II: \
A theory of traffic flow on long crowded roads\", ",
   StyleBox["Proc. Roy. Soc. A", FontSlant -> Italic],
   " 229:317\[Dash]345.\n\[Bullet] ",
   "M. Treiber, A. Hennecke & D. Helbing (2000), \"Congested traffic \
states in empirical observations and microscopic simulations\", ",
   StyleBox["Phys. Rev. E", FontSlant -> Italic],
   " 62:1805.\n\[Bullet] ",
   "M. Treiber & A. Kesting (2013), ",
   StyleBox["Traffic Flow Dynamics: Data, Models and Simulation",
     FontSlant -> Italic],
   ", Springer \[LongDash] the canonical modern textbook on the \
whole hierarchy.\n\[Bullet] ",
   "K. Nagel & M. Schreckenberg (1992), \"A cellular automaton model \
for freeway traffic\", ",
   StyleBox["J. Phys. I France", FontSlant -> Italic],
   " 2:2221.\n\[Bullet] ",
   "M. Bando, K. Hasebe, A. Nakayama, A. Shibata & Y. Sugiyama \
(1995), \"Dynamical model of traffic congestion and numerical \
simulation\", ",
   StyleBox["Phys. Rev. E", FontSlant -> Italic],
   " 51:1035.\n\[Bullet] ",
   "J. G. Wardrop (1952), \"Some theoretical aspects of road traffic \
research\", ",
   StyleBox["Proc. ICE", FontSlant -> Italic],
   " 1(3):325\[Dash]378.\n\[Bullet] ",
   "M. J. Beckmann, C. B. McGuire & C. B. Winsten (1956), ",
   StyleBox["Studies in the Economics of Transportation",
     FontSlant -> Italic],
   ", Yale U.P.\n\[Bullet] ",
   "F. V. Webster (1958), \"Traffic signal settings\", \
HMSO Road Research Technical Paper 39.\n\[Bullet] ",
   "Source code, data, and more animations: ",
   ButtonBox["github.com/mthiel74/TrafficJAms",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://github.com/mthiel74/TrafficJAms"],
                    None}],
   "."}],
      "Text"]

};

nb = Notebook[cells,
  WindowSize -> {1100, 800},
  WindowMargins -> {{Automatic, 0}, {Automatic, 0}},
  FrontEndVersion -> "15.0 for Mac OS X ARM (64-bit) (October 27, 2025)",
  StyleDefinitions -> "Default.nb",
  Saveable -> True];

outNb  = FileNameJoin[{hereDir, "trafficjams.nb"}];
outPdf = FileNameJoin[{hereDir, "trafficjams.pdf"}];

Print["writing notebook -> ", outNb];
Export[outNb, nb];

Print["writing pdf      -> ", outPdf];
Export[outPdf, nb];

Print["done; nb=", FileByteCount[outNb],
      "  pdf=", FileByteCount[outPdf]];
