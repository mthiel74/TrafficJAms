#!/usr/bin/env wolframscript
(* Build Wolfram/Community/trafficjams.nb -- a Wolfram Notebook for the
   Wolfram Community.

   The notebook shows, for every section:
     1. Explanation text and mathematical derivation (typeset natively).
     2. An Input cell containing the exact Wolfram-Language code that
        produces the figure / animation.
     3. An Output cell with the pre-rendered result (image or
        ListAnimate with embedded frames) so the reader sees the
        output immediately, without evaluating.

   If the reader re-evaluates the Input cell, a fresh result replaces
   the pre-rendered one. The .wl packages live alongside the notebook
   (attached as trafficjams-packages.zip on the Community post). *)

hereDir = DirectoryName[If[$ScriptCommandLine =!= {}, $InputFileName,
                           NotebookFileName[]]];
resDir  = FileNameJoin[{hereDir, "..", "results"}];

imgWidth = 520;

(* We don't re-run the sims here -- all outputs come from pre-
   generated PNGs and GIFs in ../results/. The Input cells in the
   notebook re-run them on demand when the reader evaluates them. *)

(* ---------- math typesetting helpers ---------- *)

mathI[box_] := Cell[BoxData[FormBox[box, TraditionalForm]],
  "InlineFormula", FontFamily -> "Times"];
mathD[box_] := Cell[BoxData[FormBox[box, TraditionalForm]],
  "DisplayFormula", TextAlignment -> Center, FontFamily -> "Times"];

(* Input cell (code the reader can re-evaluate) *)
inputCell[str_String] := Cell[str, "Input", CellTags -> {"runnable"}];

(* Output cell: embed a Wolfram expression pre-evaluated at build time. *)
outputCell[expr_] := Cell[BoxData[ToBoxes[expr]],
  "Output", ShowCellLabel -> False,
  TextAlignment -> Center,
  CellMargins -> {{20, 20}, {10, 14}}];

(* Per-animation frame-cap and playback-rate overrides *)
$animFramesOverride = <|
  "anim_bando.gif" -> 80,
  "anim_nasch.gif" -> 40,
  "aberdeen_full.gif" -> 30,
  "aberdeen_city.gif" -> 30
|>;
$animRateOverride = <|
  "anim_nasch.gif" -> 2,
  "anim_nasch_2lane.gif" -> 3
|>;

loadAnim[name_] := With[
  {p = FileNameJoin[{resDir, name}]},
  If[!FileExistsQ[p],
    Style["[missing " <> name <> "]", Red],
    With[{all = Import[p, {"GIF", "ImageList"}],
          cap = Lookup[$animFramesOverride, name, 30],
          rate = Lookup[$animRateOverride, name, 4]},
      With[{stride = Max[1, Ceiling[Length[all]/cap]]},
        ListAnimate[
          ImageResize[#, {420}] & /@ all[[1 ;; -1 ;; stride]],
          AnimationRate -> rate,
          AnimationRunning -> True,
          AnimationRepetitions -> Infinity]]]]];

loadPng[name_] := With[
  {p = FileNameJoin[{resDir, name}]},
  If[FileExistsQ[p],
    Image[Import[p], ImageSize -> imgWidth],
    Style["[missing " <> name <> "]", Red]]];

(* All output cells are populated from pre-generated assets in
   ../results/. The reader can re-run the Input cells to regenerate
   them from scratch if they want. *)

lwrPlot   = loadPng["lwr.png"];
lwrAnim   = loadAnim["anim_lwr.gif"];
pwAnim    = loadAnim["anim_payne_whitham.gif"];
idmAnim   = loadAnim["anim_idm.gif"];
bandoAnim = loadAnim["anim_bando.gif"];
nsAnim    = loadAnim["anim_nasch.gif"];
ns2Plot   = loadPng["nasch_2lane.png"];
qPlot     = loadPng["queueing.png"];
soPlot    = loadPng["signal_optimisation.png"];
icPlot    = loadPng["signal_schedule.png"];
naPlot    = loadPng["network_assignment.png"];
abPlot    = loadPng["aberdeen_network.png"];
daPlot    = loadPng["dynamic_assignment.png"];

(* ---------- reusable math fragments (same as before) ---------- *)

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

lwrEq = RowBox[{
  FractionBox[RowBox[{"\[PartialD]", "\[Rho]"}], RowBox[{"\[PartialD]", "t"}]],
  "+", FractionBox[RowBox[{"\[PartialD]", QofRho}], RowBox[{"\[PartialD]", "x"}]],
  "=", "0"}];

greenshieldsEq = RowBox[{QofRho, "=",
  RowBox[{"\[Rho]", "\[ThinSpace]", vMaxSym,
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

idmAccEq = RowBox[{OverscriptBox[vAlpha, "\[CenterDot]"], "=",
  RowBox[{"a", " ", RowBox[{"[", RowBox[{"1", "-",
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
equilibrium assignment \[LongDash] a self-contained port of \
classical traffic-flow models."}], "Subtitle"],

 Cell[TextData[{"Marco Thiel, April 2026 \[LongDash] ",
    ButtonBox["github.com/mthiel74/TrafficJAms",
      BaseStyle -> "Hyperlink",
      ButtonData -> {URL["https://github.com/mthiel74/TrafficJAms"],
                     None}]}],
      "Text", TextAlignment -> Center, FontSlant -> Italic],

 Cell["Getting started", "Section"],

 Cell[TextData[{"Every figure below is already rendered, so you can \
read the notebook end-to-end without evaluating anything. If you do \
want to rerun or tweak the simulations, the Input cell above each \
figure contains the exact call. All the packages the code needs \
ship with this post as ",
   StyleBox["trafficjams-packages.zip", "Input"],
   " \[LongDash] download it, unzip it into the ",
   StyleBox["same directory as this notebook", FontSlant -> Italic],
   ", and then evaluate the next cell once:"}], "Text"],

 inputCell["SetDirectory[NotebookDirectory[]];
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
Get[\"CoreAnimations.wl\"];"],

 Cell[TextData[{"The full source (including two extra OSM-based \
Aberdeen simulations that aren't bundled here) is on GitHub: ",
   ButtonBox["github.com/mthiel74/TrafficJAms",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://github.com/mthiel74/TrafficJAms"],
                    None}], "."}], "Text"],

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
walks through all of those in turn, each with a short derivation, \
runnable Wolfram-Language code, and a pre-rendered result. The \
running case study is Aberdeen in north-east Scotland \[LongDash] \
a city whose peculiar geography makes it an unusually nice \
traffic-engineering laboratory."}], "Text"],

 Cell["1. Why model traffic?", "Section"],
 Cell[TextData[{"Traffic engineers need quantitative tools to answer \
questions like: What happens to queue lengths on Union Street if we \
add a bus gate? If the Bridge of Don closes for maintenance, does \
Anderson Drive absorb the flow, or does half the city seize up? None \
of these has a closed-form answer, but all of them are tractable in \
simulation once the correct mathematical model has been picked for \
the right spatial scale."}], "Text"],

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

 inputCell["lwrRes = LWR`SimulateLWR[];
LWR`PlotLWR[lwrRes]"],
 outputCell[lwrPlot],

 Cell[TextData[{"Animated version, showing the jam block disperse \
into a backward shock and a forward rarefaction fan:"}], "Text"],
 inputCell["lwrFrames = Table[CoreAnimations`LWRCorridorFrame[lwrRes, k],
  {k, 1, Length[lwrRes[\"t\"]], 2}];
ListAnimate[lwrFrames, AnimationRate -> 4]"],
 outputCell[lwrAnim],

 Cell["3. Adding a second equation: Payne-Whitham", "Section"],
 Cell[TextData[{"A weakness of LWR is that velocity is always the \
equilibrium value on the fundamental diagram. Payne (1971) and \
Whitham (1974) independently proposed a momentum equation in which \
velocity relaxes with time constant ", mathI[tauSym],
   " towards its equilibrium ", mathI[VeSym],
   " and is pushed by a pressure-like anticipation term:"}], "Text"],
 mathD[payneEq],

 inputCell["pwRes = PayneWhitham`SimulatePW[];
pwFrames = Table[CoreAnimations`PWCorridorFrame[pwRes, k],
  {k, 1, Length[pwRes[\"t\"]], 2}];
ListAnimate[pwFrames, AnimationRate -> 4]"],
 outputCell[pwAnim],

 Cell["4. The microscopic view: Intelligent Driver Model", "Section"],
 Cell[TextData[{"The Intelligent Driver Model (Treiber, Hennecke & \
Helbing, 2000) is a second-order ODE for each vehicle:"}], "Text"],
 mathD[idmAccEq],
 Cell[TextData[{"where ", mathI[vAlpha], " and ", mathI[sAlpha],
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
idmFrames = Table[
  CoreAnimations`IDMRingFrame[idmRes, k, 20],
  {k, 1, Length[idmRes[\"times\"]], 5}];
ListAnimate[idmFrames, AnimationRate -> 4]"],
 outputCell[idmAnim],

 Cell[TextData[{"This is exactly the phenomenon Sugiyama et al. (2008) \
demonstrated experimentally on a real circular track \[LongDash] \
their striking video is worth watching alongside the simulation: ",
   ButtonBox["youtube.com/watch?v=Rryu85BtALM",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://www.youtube.com/watch?v=Rryu85BtALM"],
                    None}], "."}], "Text"],

 Cell["5. The simplest car-following that jams: Bando OVM", "Section"],
 Cell[TextData[{"Even simpler than IDM, the Optimal Velocity Model \
(Bando et al. 1995) relaxes speed towards a target velocity that \
depends only on the gap to the vehicle ahead:"}], "Text"],
 mathD[bandoEq],
 Cell[TextData[{"with the standard choice"}], "Text"],
 mathD[bandoVopt],
 Cell[TextData[{"Stability parameter ", mathI[kappa],
   ": with ", mathI[RowBox[{kappa, "=0.4"}]],
   " below the stability threshold on an 800 m ring of 18 vehicles, \
the Hopf instability grows over the 2000 s run:"}], "Text"],

 inputCell["bandoRes = Bando`SimulateBando[
  \"nVehicles\" -> 18, \"T\" -> 2000.0,
  \"roadLength\" -> 800.0, \"kappa\" -> 0.4];
bandoFrames = Table[
  CoreAnimations`BandoRingFrame[bandoRes, k, 12],
  {k, 1, Length[bandoRes[\"times\"]], 5}];
ListAnimate[bandoFrames, AnimationRate -> 4]"],
 outputCell[bandoAnim],

 Cell["6. A stochastic cellular automaton: Nagel-Schreckenberg",
   "Section"],
 Cell[TextData[{"Nagel and Schreckenberg (1992) gave the minimal CA \
that reproduces realistic traffic dynamics. Space is 7.5 m cells, \
time is 1 s steps, velocity is an integer in [0, ",
   mathI[vMaxCell], "]. Each step applies four rules:"}], "Text"],
 mathD[naschRule1], mathD[naschRule2], mathD[naschRule3], mathD[naschRule4],

 inputCell["nsRes = NagelSchreckenberg`SimulateNaSch[
  \"T\" -> 300, \"nVehicles\" -> 80, \"roadLength\" -> 300];
nsFrames = Table[
  CoreAnimations`NaSchHighwayFrame[nsRes, t],
  {t, 1, nsRes[\"T\"], 5}];
ListAnimate[nsFrames, AnimationRate -> 2]"],
 outputCell[nsAnim],

 Cell[TextData[{"Extending to two lanes with a symmetric lane-change \
rule recovers the classic multi-lane redistribution:"}], "Text"],
 inputCell["ns2Res = NagelSchreckenberg2Lane`SimulateNaSch2[
  \"T\" -> 200, \"nVehicles\" -> 150, \"roadLength\" -> 500];
NagelSchreckenberg2Lane`PlotNaSch2[ns2Res]"],
 outputCell[ns2Plot],

 Cell["7. Intersections as queues: M/D/1", "Section"],
 Cell[TextData[{"With Poisson arrivals at rate ", mathI[lambda],
   " and deterministic service at rate ", mathI[mu],
   ", the Pollaczek\[Dash]Khinchine formula for the M/D/1 queue:"}],
   "Text"],
 mathD[md1LqEq], mathD[md1WqEq],
 Cell[TextData[{"both diverging as utilisation ", mathI[rhoDef],
   " approaches 1."}], "Text"],
 inputCell["qRes = Queueing`SimulateQueueing[];
Queueing`PlotQueueing[qRes]"],
 outputCell[qPlot],

 Cell["8. Optimising the signal timing", "Section"],
 Cell[TextData[{"The M/D/1 waiting term plus Webster's uniform \
delay:"}], "Text"],
 mathD[websterEq],
 Cell[TextData[{"gives a smooth convex objective minimised by ",
   StyleBox["FindMinimum", "Input"], " in milliseconds:"}], "Text"],
 inputCell["soRes = SignalOptimisation`SimulateSignalOpt[];
SignalOptimisation`PlotSignalOpt[soRes]"],
 outputCell[soPlot],
 Cell[TextData[{"Signal controller (90 s cycle, 4 s amber, \
round-robin phases):"}], "Text"],
 inputCell["icRes = IntersectionControl`SimulateSignalDemo[];
IntersectionControl`PlotSignalDemo[icRes]"],
 outputCell[icPlot],

 Cell["9. Network-level assignment: Wardrop equilibrium", "Section"],
 Cell[TextData[{"Wardrop's first principle (1952): at equilibrium \
all used paths between an OD pair have equal cost:"}], "Text"],
 mathD[wardropEq],
 Cell[TextData[{"equivalently (Beckmann 1956) the unique minimiser \
of"}], "Text"],
 mathD[beckmannEq],
 Cell[TextData[{"with BPR link costs"}], "Text"],
 mathD[bprEq],
 inputCell["naRes = NetworkAssignment`SimulateAssignment[];
NetworkAssignment`PlotAssignment[naRes]"],
 outputCell[naPlot],
 Cell[TextData[{"On the curated 15-node Aberdeen network:"}], "Text"],
 inputCell["abRes = AberdeenNetwork`SimulateAberdeenNetwork[];
AberdeenNetwork`PlotAberdeenNetwork[abRes]"],
 outputCell[abPlot],
 Cell[TextData[{"Union Street and Anderson Drive (south) emerge as \
the dominant bottlenecks \[LongDash] saturation 3.0\[Times] and \
2.4\[Times] respectively."}], "Text"],

 Cell["10. Time-varying demand", "Section"],
 Cell[TextData[{"Running Wardrop period by period with residual \
congestion carry-over:"}], "Text"],
 inputCell["daRes = DynamicAssignment`SimulateDynamicAssignment[];
DynamicAssignment`PlotDynamicAssignment[daRes]"],
 outputCell[daPlot],

 Cell["11. OSM-based Aberdeen multi-agent simulations", "Section"],
 Cell[TextData[{"The GitHub repo additionally ships two larger \
simulations that need OpenStreetMap JSON caches (omitted here to \
keep the bundle small): a 1397-node city-centre multi-agent sim \
(",
   StyleBox["AberdeenCityNetwork.wl", "Input"],
   ") and a 11\[ThinSpace]925-node full-city v2 with traffic signals \
and roundabout yield (",
   StyleBox["AberdeenFullCity.wl", "Input"],
   "). For reference, here are the pre-rendered animations from the \
repo:"}], "Text"],
 outputCell[loadAnim["aberdeen_city.gif"]],
 outputCell[loadAnim["aberdeen_full.gif"]],

 Cell["12. Implementation notes", "Section"],
 Cell[TextData[{
   "\[Bullet] Each model is a standalone .wl package with a ",
   StyleBox["Simulate<Name>[opts]", "Input"],
   " entry point returning an Association and a matching ",
   StyleBox["Plot<Name>[result]", "Input"], " renderer.\n",
   "\[Bullet] Car-following loops (IDM, Bando) use vectorised ",
   StyleBox["RotateLeft", "Input"],
   " to compute leader-follower gaps in one call.\n",
   "\[Bullet] Both network-assignment models use ",
   StyleBox["FindMinimum[{obj, constraints}, vars, Method -> \
\"InteriorPoint\"]", "Input"],
   " on the Beckmann integral.\n",
   "\[Bullet] Animations are plain ",
   StyleBox["ListAnimate", "Input"],
   "s over frame lists built by the functions in ",
   StyleBox["CoreAnimations.wl", "Input"], "."}], "Text"],

 Cell["13. References", "Section"],
 Cell[TextData[{
   "\[Bullet] M. J. Lighthill & G. B. Whitham (1955), \"On kinematic \
waves II\", ", StyleBox["Proc. Roy. Soc. A", FontSlant -> Italic],
   " 229:317.\n\[Bullet] ",
   "M. Treiber, A. Hennecke & D. Helbing (2000), \"Congested traffic \
states in empirical observations and microscopic simulations\", ",
   StyleBox["Phys. Rev. E", FontSlant -> Italic], " 62:1805.\n\[Bullet] ",
   "M. Treiber & A. Kesting (2013), ",
   StyleBox["Traffic Flow Dynamics", FontSlant -> Italic], ", Springer.\n",
   "\[Bullet] K. Nagel & M. Schreckenberg (1992), ",
   StyleBox["J. Phys. I France", FontSlant -> Italic],
   " 2:2221.\n\[Bullet] M. Bando et al. (1995), ",
   StyleBox["Phys. Rev. E", FontSlant -> Italic],
   " 51:1035.\n\[Bullet] J. G. Wardrop (1952), ",
   StyleBox["Proc. ICE", FontSlant -> Italic],
   " 1(3):325.\n\[Bullet] Y. Sugiyama et al. (2008), ",
   StyleBox["New J. Phys.", FontSlant -> Italic],
   " 10:033001. Video: ",
   ButtonBox["youtube.com/watch?v=Rryu85BtALM",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://www.youtube.com/watch?v=Rryu85BtALM"],
                    None}],
   ".\n\[Bullet] Source: ",
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

outNb = FileNameJoin[{hereDir, "trafficjams.nb"}];
Print["writing notebook -> ", outNb];
Export[outNb, nb];
Print["done; nb=", FileByteCount[outNb]];
