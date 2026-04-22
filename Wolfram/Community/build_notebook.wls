#!/usr/bin/env wolframscript
(* Build Wolfram/Community/trafficjams.nb as a code-first interactive
   Wolfram Notebook suitable for posting to the Wolfram Community with
   the accompanying .wl packages attached.

   Every figure and animation is (re)generated at evaluation time by
   calling the functions in IDM.wl, Bando.wl, ... etc.  The notebook
   expects the .wl files to sit in the same directory as itself; the
   first input cell takes care of setting that up via
   SetDirectory[NotebookDirectory[]]. *)

hereDir = DirectoryName[If[$ScriptCommandLine =!= {}, $InputFileName,
                           NotebookFileName[]]];

(* ---------- math typesetting helpers (same as before) ---------- *)

mathI[box_] := Cell[BoxData[FormBox[box, TraditionalForm]],
  "InlineFormula", FontFamily -> "Times"];
mathD[box_] := Cell[BoxData[FormBox[box, TraditionalForm]],
  "DisplayFormula", TextAlignment -> Center, FontFamily -> "Times"];

(* Input cell that the reader can re-evaluate *)
inputCell[str_String] := Cell[str, "Input"];
inputCell[cellContent_BoxData] := Cell[cellContent, "Input"];

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

(* ---------- display equations (same as pre-rendered version) ---------- *)

lwrEq = RowBox[{
  FractionBox[RowBox[{"\[PartialD]", "\[Rho]"}], RowBox[{"\[PartialD]", "t"}]],
  "+",
  FractionBox[RowBox[{"\[PartialD]", QofRho}], RowBox[{"\[PartialD]", "x"}]],
  "=", "0"}];

greenshieldsEq = RowBox[{
  QofRho, "=", RowBox[{"\[Rho]", "\[ThinSpace]", vMaxSym,
    RowBox[{"(", RowBox[{"1", "-", FractionBox["\[Rho]", rhoMax]}], ")"}]}]}];

pwLHS = RowBox[{
  FractionBox[RowBox[{"\[PartialD]", "v"}], RowBox[{"\[PartialD]", "t"}]],
  "+", "v",
  FractionBox[RowBox[{"\[PartialD]", "v"}], RowBox[{"\[PartialD]", "x"}]]}];
payneEq = RowBox[{pwLHS, "=",
  FractionBox[RowBox[{VeSym, "-", "v"}], tauSym], "-",
  FractionBox[SuperscriptBox[c0Sym, "2"], "\[Rho]"],
  FractionBox[RowBox[{"\[PartialD]", "\[Rho]"}], RowBox[{"\[PartialD]", "x"}]]
}];

idmAccEq = RowBox[{
  OverscriptBox[vAlpha, "\[CenterDot]"], "=",
  RowBox[{"a", " ",
    RowBox[{"[", RowBox[{"1", "-",
      SuperscriptBox[RowBox[{"(", FractionBox[vAlpha, v0Sym], ")"}], "\[Delta]"],
      "-", SuperscriptBox[RowBox[{"(",
        FractionBox[RowBox[{sStar, "(", vAlpha, ",", " ", dvAlpha, ")"}],
          sAlpha], ")"}], "2"]}], "]"}]}]}];

idmGapEq = RowBox[{RowBox[{sStar, "(", "v", ",", " ", "\[CapitalDelta]v", ")"}],
  "=", RowBox[{s0Sym, "+", RowBox[{"v", "\[ThinSpace]", THead}], "+",
    FractionBox[RowBox[{"v", "\[ThinSpace]", "\[CapitalDelta]v"}],
      RowBox[{"2", SqrtBox[RowBox[{"a", "\[ThinSpace]", "b"}]]}]]}]}];

bandoEq = RowBox[{OverscriptBox[vAlpha, "\[CenterDot]"], "=",
  RowBox[{kappa, " ", RowBox[{"[", RowBox[{Vopt, "(", sAlpha, ")", "-", vAlpha}], "]"}]}]}];

bvInner = RowBox[{"(", RowBox[{"tanh", "(",
    RowBox[{FractionBox["s", scSym], "-", "2"}], ")"}], "+",
  RowBox[{"tanh", "(", "2", ")"}], ")"}];
bandoVopt = RowBox[{RowBox[{Vopt, "(", "s", ")"}], "=",
  FractionBox[RowBox[{vMaxOVM, "\[ThinSpace]", bvInner}],
    RowBox[{"1", "+", RowBox[{"tanh", "(", "2", ")"}]}]]}];

naschRule1 = RowBox[{"(", "1", ")", " ", "accelerate:", " ",
  "v", "\[Rule]", RowBox[{"min", "(", RowBox[{"v", "+", "1"}], ",", " ", vMaxCell, ")"}]}];
naschRule2 = RowBox[{"(", "2", ")", " ", "brake:", " ",
  "v", "\[Rule]", RowBox[{"min", "(", "v", ",", " ", RowBox[{"d", "-", "1"}], ")"}]}];
naschRule3 = RowBox[{"(", "3", ")", " ", "randomise:", " ",
  RowBox[{"with", " ", "probability", " ", pSlow}], ",", " ",
  "v", "\[Rule]", RowBox[{"max", "(", RowBox[{"v", "-", "1"}], ",", " ", "0", ")"}]}];
naschRule4 = RowBox[{"(", "4", ")", " ", "move:", " ", "x", "\[Rule]", RowBox[{"x", "+", "v"}]}];

md1LqEq = RowBox[{LqSym, "=", FractionBox[SuperscriptBox[rhoSym, "2"],
  RowBox[{"2", "(", RowBox[{"1", "-", rhoSym}], ")"}]]}];
md1WqEq = RowBox[{WqSym, "=", FractionBox[rhoSym,
  RowBox[{"2", "\[ThinSpace]", mu, "\[ThinSpace]",
    RowBox[{"(", RowBox[{"1", "-", rhoSym}], ")"}]}]]}];
rhoDef  = RowBox[{rhoSym, "=", FractionBox[lambda, mu]}];

bprEq = RowBox[{teX, "=", RowBox[{te0, " ",
  RowBox[{"[", RowBox[{"1", "+",
    RowBox[{alphaSym, " ", SuperscriptBox[
      RowBox[{"(", FractionBox[xeSym, CeSym], ")"}], betaSym]}]}], "]"}]}]}];

beckmannEq = RowBox[{RowBox[{UnderscriptBox["min", "x"]}], " ", RowBox[{
  UnderoverscriptBox["\[Sum]", RowBox[{"e", "\[Element]", "E"}], ""], " ",
  SubsuperscriptBox["\[Integral]", "0", xeSym], " ",
  RowBox[{SubscriptBox["t", "e"], "(", "\[Omega]", ")"}],
  RowBox[{"\[DifferentialD]", "\[Omega]"}]}]}];

wardropEq = RowBox[{SubscriptBox["c", "p"], "=", SubscriptBox["c", "q"],
  "\[ThickSpace]\[ThickSpace]\[ThickSpace]",
  RowBox[{"\[ForAll]", " ", "used", " ", "paths", " ", "p", ",", " ", "q"}]}];

websterEq = RowBox[{"d", SubscriptBox["", "unif"], "=",
  FractionBox[SuperscriptBox[webRed, "2"], RowBox[{"2", Cycle}]]}];

(* ------------------------------------------------------------------ *)

cells = {

 Cell["Traffic Flow Dynamics in the Wolfram Language", "Title"],

 Cell[TextData[{"From phantom jams on a ring road to Wardrop-\
equilibrium assignment \[LongDash] a self-contained, runnable \
port of classical traffic-flow models."}], "Subtitle"],

 Cell[TextData[{"Marco Thiel, April 2026 \[LongDash] ",
    ButtonBox["github.com/mthiel74/TrafficJAms",
      BaseStyle -> "Hyperlink",
      ButtonData -> {URL["https://github.com/mthiel74/TrafficJAms"],
                     None}]}],
      "Text", TextAlignment -> Center, FontSlant -> Italic],

 (* ---- Getting started ---- *)

 Cell["Getting started", "Section"],

 Cell[TextData[{"This notebook is code-first: every figure and \
animation is (re)generated at evaluation time by the packages \
shipped alongside this post. Make sure all the ",
   StyleBox[".wl", "Input"],
   " attachments (",
   StyleBox["IDM.wl", "Input"], ", ",
   StyleBox["Bando.wl", "Input"], ", ",
   StyleBox["LWR.wl", "Input"], ", ",
   StyleBox["PayneWhitham.wl", "Input"], ", ",
   StyleBox["NagelSchreckenberg.wl", "Input"], ", ",
   StyleBox["NagelSchreckenberg2Lane.wl", "Input"], ", ",
   StyleBox["Queueing.wl", "Input"], ", ",
   StyleBox["SignalOptimisation.wl", "Input"], ", ",
   StyleBox["IntersectionControl.wl", "Input"], ", ",
   StyleBox["NetworkAssignment.wl", "Input"], ", ",
   StyleBox["DynamicAssignment.wl", "Input"], ", ",
   StyleBox["AberdeenNetwork.wl", "Input"], ", ",
   StyleBox["CoreAnimations.wl", "Input"],
   ") sit in the same directory as this notebook, then evaluate the \
next cell. All subsequent sections will then be self-contained."}],
      "Text"],

 inputCell["SetDirectory[NotebookDirectory[]];
(* Load all model packages *)
Get[\"IDM.wl\"];
Get[\"Bando.wl\"];
Get[\"LWR.wl\"];
Get[\"PayneWhitham.wl\"];
Get[\"NagelSchreckenberg.wl\"];
Get[\"NagelSchreckenberg2Lane.wl\"];
Get[\"Queueing.wl\"];
Get[\"SignalOptimisation.wl\"];
Get[\"IntersectionControl.wl\"];
Get[\"NetworkAssignment.wl\"];
Get[\"DynamicAssignment.wl\"];
Get[\"AberdeenNetwork.wl\"];
Get[\"CoreAnimations.wl\"];
\"All packages loaded.\""],

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
walks through all of those in turn, each with a short derivation and \
runnable Wolfram-Language implementation. The running case study is \
Aberdeen in north-east Scotland \[LongDash] a city whose peculiar \
geography (squeezed between the Rivers Don and Dee, the coast, and \
the hills, with two critical bridges) makes it an unusually nice \
traffic-engineering laboratory."}], "Text"],

 (* ---- Section 1 ---- *)

 Cell["1. Why model traffic?", "Section"],

 Cell[TextData[{
   "Traffic engineers need quantitative tools to answer questions \
like: What happens to queue lengths on Union Street if we add a bus \
gate? If the Bridge of Don closes for maintenance, does Anderson \
Drive absorb the flow, or does half the city seize up? None of these \
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
model for it, and lets you run it."}], "Text"],

 (* ---- Section 2: LWR ---- *)

 Cell["2. The macroscopic view: the LWR model", "Section"],

 Cell[TextData[{"Lighthill and Whitham (1955), followed independently \
by Richards (1956), proposed that traffic on a long road obeys the \
same conservation law as any other compressible one-dimensional \
fluid: vehicles cannot be created or destroyed, so the density ",
   mathI[rhoXT], " and flow ", mathI[QofRho], " satisfy"}], "Text"],

 mathD[lwrEq],

 Cell[TextData[{"The missing ingredient is a ",
   StyleBox["fundamental diagram", FontSlant -> Italic],
   ". The simplest, due to Greenshields (1935), is linear in the \
speed:"}], "Text"],

 mathD[greenshieldsEq],

 Cell[TextData[{"Plug this into the conservation law, discretise \
with the Godunov upwind flux, and you get a handful of lines of code:"}],
      "Text"],

 inputCell["lwrRes = LWR`SimulateLWR[];
LWR`PlotLWR[lwrRes]"],

 Cell[TextData[{"An animated version of the same run, seeded by a \
high-density block in the middle of the road:"}], "Text"],

 inputCell["lwrFrames = Table[CoreAnimations`LWRCorridorFrame[lwrRes, k],
   {k, 1, Length[lwrRes[\"t\"]], 2}];
ListAnimate[lwrFrames, AnimationRate -> 4,
   AnimationRunning -> True,
   AnimationRepetitions -> Infinity]"],

 (* ---- Section 3: Payne-Whitham ---- *)

 Cell["3. Adding a second equation: Payne-Whitham", "Section"],

 Cell[TextData[{"A weakness of LWR is that velocity is always the \
equilibrium value on the fundamental diagram. Payne (1971) and \
Whitham (1974) independently proposed a momentum equation in which \
velocity relaxes with time constant ", mathI[tauSym],
   " towards its equilibrium ", mathI[VeSym],
   " and is pushed by a pressure-like anticipation term:"}], "Text"],

 mathD[payneEq],

 inputCell["pwRes = PayneWhitham`SimulatePW[];
PayneWhitham`PlotPW[pwRes]"],

 inputCell["pwFrames = Table[CoreAnimations`PWCorridorFrame[pwRes, k],
   {k, 1, Length[pwRes[\"t\"]], 2}];
ListAnimate[pwFrames, AnimationRate -> 4,
   AnimationRunning -> True,
   AnimationRepetitions -> Infinity]"],

 (* ---- Section 4: IDM ---- *)

 Cell["4. The microscopic view: Intelligent Driver Model", "Section"],

 Cell[TextData[{"The Intelligent Driver Model (Treiber, Hennecke & \
Helbing, 2000) is a second-order ODE for the longitudinal dynamics \
of each vehicle:"}], "Text"],

 mathD[idmAccEq],

 Cell[TextData[{"where ",
   mathI[vAlpha], " and ", mathI[sAlpha],
   " are the speed and gap to the vehicle in front, ", mathI[dvAlpha],
   " is the approach rate, and the desired gap is"}], "Text"],

 mathD[idmGapEq],

 Cell[TextData[{"The parameters below put the system firmly in the \
stop-and-go regime: a 25-vehicle 1 km ring with ",
   mathI[RowBox[{v0Sym, "=30"}]], " m/s desired speed, ",
   mathI[RowBox[{"T", "=2.5"}]], " s headway, sluggish acceleration ",
   mathI[RowBox[{"a", "=0.3"}]], " m/",
   mathI[SuperscriptBox["s", "2"]], " and firm braking ",
   mathI[RowBox[{"b", "=2.0"}]], " m/", mathI[SuperscriptBox["s", "2"]],
   ". One vehicle starts at half speed; the instability amplifies \
into a persistent backward-propagating jam."}], "Text"],

 inputCell["idmRes = IDM`SimulateIDM[
  \"nVehicles\" -> 25, \"T\" -> 400.0, \"roadLength\" -> 1000.0,
  \"v0\" -> 30.0, \"s0\" -> 2.0, \"THeadway\" -> 2.5,
  \"a\" -> 0.3, \"b\" -> 2.0];
idmFrames = Table[CoreAnimations`IDMRingFrame[idmRes, k, 20],
   {k, 1, Length[idmRes[\"times\"]], 5}];
ListAnimate[idmFrames, AnimationRate -> 4,
   AnimationRunning -> True,
   AnimationRepetitions -> Infinity]"],

 Cell[TextData[{
   "This is exactly the same phenomenon that Sugiyama et al. (2008) \
demonstrated experimentally on a real circular track \[LongDash] \
their striking video is worth watching alongside the simulation: ",
   ButtonBox["youtube.com/watch?v=Rryu85BtALM",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://www.youtube.com/watch?v=Rryu85BtALM"],
                    None}], "."}], "Text"],

 (* ---- Section 5: Bando ---- *)

 Cell["5. The simplest car-following that jams: Bando OVM", "Section"],

 Cell[TextData[{"Even simpler than IDM, the Optimal Velocity Model \
(Bando et al. 1995) relaxes speed towards a target velocity that \
depends only on the gap to the vehicle ahead:"}], "Text"],

 mathD[bandoEq],

 Cell[TextData[{"with the standard choice"}], "Text"],

 mathD[bandoVopt],

 Cell[TextData[{"The model has one stability parameter ",
   mathI[kappa], ". For ", mathI[kappa],
   " above a critical value the uniform flow is stable; below it, \
uniform flow is linearly unstable via a Hopf bifurcation. With ",
   mathI[RowBox[{kappa, "=0.4"}]], " on an 800 m ring of 18 vehicles \
we are firmly in the unstable regime \[LongDash] watch the speed \
spread widen over the 2000 s run:"}], "Text"],

 inputCell["bandoRes = Bando`SimulateBando[
  \"nVehicles\" -> 18, \"T\" -> 2000.0, \"roadLength\" -> 800.0,
  \"kappa\" -> 0.4];
bandoFrames = Table[CoreAnimations`BandoRingFrame[bandoRes, k, 12],
   {k, 1, Length[bandoRes[\"times\"]], 5}];
ListAnimate[bandoFrames, AnimationRate -> 4,
   AnimationRunning -> True,
   AnimationRepetitions -> Infinity]"],

 (* ---- Section 6: NaSch ---- *)

 Cell["6. A stochastic cellular automaton: Nagel-Schreckenberg", "Section"],

 Cell[TextData[{"Nagel and Schreckenberg (1992) gave the minimal CA \
that reproduces realistic traffic dynamics. Space is discretised \
into 7.5 m cells, time into 1 s steps, and velocity is an integer in \
[0, ", mathI[vMaxCell], "]. At each step every vehicle applies four \
rules:"}], "Text"],

 mathD[naschRule1],
 mathD[naschRule2],
 mathD[naschRule3],
 mathD[naschRule4],

 Cell[TextData[{"where ", mathI["d"],
   " is the number of empty cells to the next vehicle. The stochastic \
slow-down rule is essential: without it the model is deterministic \
and does not jam spontaneously."}], "Text"],

 inputCell["nsRes = NagelSchreckenberg`SimulateNaSch[
   \"T\" -> 300, \"nVehicles\" -> 80, \"roadLength\" -> 300];
nsFrames = Table[CoreAnimations`NaSchHighwayFrame[nsRes, t],
   {t, 1, nsRes[\"T\"], 5}];
ListAnimate[nsFrames, AnimationRate -> 2,
   AnimationRunning -> True,
   AnimationRepetitions -> Infinity]"],

 Cell[TextData[{"Extending to two lanes with a symmetric lane-change \
rule (incentive: would have to brake in current lane; safety: \
sufficient gap ahead and behind in the target lane; target cell \
free) recovers the classic multi-lane redistribution:"}], "Text"],

 inputCell["ns2Res = NagelSchreckenberg2Lane`SimulateNaSch2[
   \"T\" -> 300, \"nVehicles\" -> 150, \"roadLength\" -> 500];
NagelSchreckenberg2Lane`PlotNaSch2[ns2Res]"],

 (* ---- Section 7: queues ---- *)

 Cell["7. Intersections as queues: M/D/1", "Section"],

 Cell[TextData[{"A signalised intersection looks like a server that \
alternates between green and red. With Poisson arrivals at rate ",
   mathI[lambda], " and deterministic service at rate ", mathI[mu],
   ", the Pollaczek\[Dash]Khinchine formula for the M/D/1 queue \
gives"}], "Text"],

 mathD[md1LqEq],
 mathD[md1WqEq],

 Cell[TextData[{"both diverging as utilisation ",
   mathI[rhoDef], " approaches 1."}], "Text"],

 inputCell["qRes = Queueing`SimulateQueueing[];
Queueing`PlotQueueing[qRes]"],

 (* ---- Section 8: signal opt ---- *)

 Cell["8. Optimising the signal timing", "Section"],

 Cell[TextData[{"Given a corridor of signals at which arrivals vary \
by intersection, what green times minimise total delay? The M/D/1 \
waiting term above plus Webster's uniform delay"}], "Text"],

 mathD[websterEq],

 Cell[TextData[{"gives a smooth convex objective that ",
   StyleBox["FindMinimum[\[CenterEllipsis], Method -> \"InteriorPoint\"]",
     "Input"],
   " solves in milliseconds:"}], "Text"],

 inputCell["soRes = SignalOptimisation`SimulateSignalOpt[];
SignalOptimisation`PlotSignalOpt[soRes]"],

 Cell[TextData[{"The signal controller itself \[LongDash] 90 s cycle, \
4 s amber, round-robin approach phases \[LongDash] is a tiny \
data-driven Association. Here is its Gantt-style schedule:"}], "Text"],

 inputCell["icRes = IntersectionControl`SimulateSignalDemo[];
IntersectionControl`PlotSignalDemo[icRes]"],

 (* ---- Section 9: network assignment ---- *)

 Cell["9. Network-level assignment: Wardrop equilibrium", "Section"],

 Cell[TextData[{"At equilibrium, all used paths between the same \
origin-destination pair have equal cost (Wardrop 1952):"}], "Text"],

 mathD[wardropEq],

 Cell[TextData[{"which is equivalent (Beckmann 1956) to the unique \
minimiser of the convex program"}], "Text"],

 mathD[beckmannEq],

 Cell[TextData[{"with the Bureau of Public Roads link-cost function"}],
      "Text"],

 mathD[bprEq],

 Cell[TextData[{"where ", mathI[RowBox[{alphaSym, "=0.15"}]], " and ",
   mathI[RowBox[{betaSym, "=4"}]],
   ". On an 8-node Aberdeen-inspired network:"}], "Text"],

 inputCell["naRes = NetworkAssignment`SimulateAssignment[];
NetworkAssignment`PlotAssignment[naRes]"],

 Cell[TextData[{"And on the curated 15-node Aberdeen network \
(Haudagain, Bridge of Don, Bridge of Dee, Union Street, Anderson \
Drive, AWPR junction, etc.) with 4 OD pairs summing to 8500 veh/h:"}],
      "Text"],

 inputCell["abRes = AberdeenNetwork`SimulateAberdeenNetwork[];
AberdeenNetwork`PlotAberdeenNetwork[abRes]"],

 Cell[TextData[{"The solver identifies Union Street and Anderson \
Drive (south) as the dominant bottlenecks \[LongDash] saturation 3.0\
\[Times] and 2.4\[Times] respectively, matching real-world \
commuter experience."}], "Text"],

 (* ---- Section 10: dynamic assignment ---- *)

 Cell["10. Time-varying demand", "Section"],

 Cell[TextData[{"Rush-hour demand is a Gaussian-ish profile. Running \
the Wardrop solve period by period, with a fraction of each period's \
flow carrying over as residual congestion that reduces the \
effective capacity for the next period:"}], "Text"],

 inputCell["daRes = DynamicAssignment`SimulateDynamicAssignment[];
DynamicAssignment`PlotDynamicAssignment[daRes]"],

 Cell[TextData[{"Flows on the shortest route pile up first, spill \
over to secondary routes as the fastest route saturates, and decay \
asymmetrically on the way out of the peak because residual \
congestion persists."}], "Text"],

 (* ---- Section 11: implementation notes ---- *)

 Cell["11. Implementation notes", "Section"],

 Cell[TextData[{
   "\[Bullet] Each model is a standalone ",
   StyleBox[".wl", "Input"],
   " package with a ", StyleBox["Simulate<Name>[opts]", "Input"],
   " entry point returning an Association, and a matching ",
   StyleBox["Plot<Name>[result]", "Input"], " renderer.\n",
   "\[Bullet] Car-following loops (IDM, Bando) use vectorised ",
   StyleBox["RotateLeft", "Input"],
   " to compute the leader\[Hyphen]follower gaps in one call.\n",
   "\[Bullet] Both network-assignment models use ",
   StyleBox["FindMinimum[{obj, constraints}, vars, Method -> \
\"InteriorPoint\"]", "Input"],
   " on the Beckmann integral.\n",
   "\[Bullet] The animations are plain ",
   StyleBox["ListAnimate", "Input"],
   "s over pre-computed frame lists; see ",
   StyleBox["CoreAnimations.wl", "Input"], " for the frame renderers.\n",
   "\[Bullet] The GitHub repo additionally ships two OSM-based \
Aberdeen simulations (city centre, 1397 nodes; full city v2, \
11\[ThinSpace]925 nodes with 151 signals and 1000 roundabout \
edges) and their animation output. They are omitted here to keep \
the notebook self-contained \[LongDash] see the ",
   ButtonBox["repository",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://github.com/mthiel74/TrafficJAms"],
                    None}],
   " for details."}], "Text"],

 (* ---- Section 12: references ---- *)

 Cell["12. References", "Section"],

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
   ", Springer.\n\[Bullet] ",
   "K. Nagel & M. Schreckenberg (1992), \"A cellular automaton model \
for freeway traffic\", ",
   StyleBox["J. Phys. I France", FontSlant -> Italic],
   " 2:2221.\n\[Bullet] ",
   "M. Bando et al. (1995), \"Dynamical model of traffic congestion \
and numerical simulation\", ",
   StyleBox["Phys. Rev. E", FontSlant -> Italic],
   " 51:1035.\n\[Bullet] ",
   "J. G. Wardrop (1952), \"Some theoretical aspects of road traffic \
research\", ",
   StyleBox["Proc. ICE", FontSlant -> Italic],
   " 1(3):325\[Dash]378.\n\[Bullet] ",
   "Y. Sugiyama et al. (2008), \"Traffic jams without bottlenecks\
\[LongDash]experimental evidence for the physical mechanism of the \
formation of a jam\", ",
   StyleBox["New J. Phys.", FontSlant -> Italic],
   " 10:033001. Video: ",
   ButtonBox["youtube.com/watch?v=Rryu85BtALM",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://www.youtube.com/watch?v=Rryu85BtALM"],
                    None}],
   ".\n\[Bullet] ",
   "Source code: ",
   ButtonBox["github.com/mthiel74/TrafficJAms",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://github.com/mthiel74/TrafficJAms"],
                    None}],
   "."}], "Text"]

};

nb = Notebook[cells,
  WindowSize -> {1100, 800},
  WindowMargins -> {{Automatic, 0}, {Automatic, 0}},
  FrontEndVersion -> "15.0 for Mac OS X ARM (64-bit) (October 27, 2025)",
  StyleDefinitions -> "Default.nb",
  Saveable -> True];

outNb  = FileNameJoin[{hereDir, "trafficjams.nb"}];

Print["writing notebook -> ", outNb];
Export[outNb, nb];
Print["done; nb=", FileByteCount[outNb]];
