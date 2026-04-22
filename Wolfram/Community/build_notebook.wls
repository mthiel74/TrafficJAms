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
Anderson Drive absorb the flow, or does half the city seize up? \
What is the shortest green phase at Mounthooly that still keeps \
average delay below 30 s? None of these has a closed-form answer, \
but all of them are tractable in simulation once the correct \
mathematical model has been picked for the right spatial scale."}],
      "Text"],

 Cell[TextData[{"Traffic is a textbook example of ",
   StyleBox["emergence", FontSlant -> Italic],
   ": individual drivers obey simple local rules (don't hit the car \
in front, change lane if you have to brake), yet the population \
collectively produces structures \[LongDash] phantom jams, \
shockwaves, stop-and-go oscillations, city-scale gridlock \
\[LongDash] that no single driver is aware of creating. This makes \
traffic a natural fit for the hierarchy of models below: a \
continuum PDE captures the coarse behaviour, an ODE per vehicle \
reveals the microscopic instability, and a directed graph plus a \
convex program describes the network-scale equilibrium. Each scale \
gets something right the others can't, and a real-world deployment \
often stitches several together."}], "Text"],

 Cell[TextData[{"Aberdeen in north-east Scotland is a good running \
example. It's small enough to be tractable (population ~200\
\[ThinSpace]000, compact historic centre), large enough to be \
interesting (international airport, oil-industry commuter flows, a \
peripheral ring road opened in 2018), and its geography creates \
natural bottlenecks \[LongDash] the Bridge of Don and Bridge of Dee, \
the Haudagain roundabout, the Anderson Drive corridor \[LongDash] \
that any model will want to reproduce."}], "Text"],

 Cell["2. The macroscopic view: the LWR model", "Section"],
 Cell[TextData[{"Lighthill and Whitham (1955), followed independently \
by Richards (1956), proposed that traffic on a long road obeys the \
same conservation law as any other compressible one-dimensional \
fluid: vehicles cannot be created or destroyed, so the density ",
   mathI[rhoXT], " and flow ", mathI[QofRho], " satisfy"}], "Text"],
 mathD[lwrEq],
 Cell[TextData[{"The missing ingredient is a ",
   StyleBox["fundamental diagram", FontSlant -> Italic],
   " \[LongDash] an empirical relation between density and flow. The \
simplest, due to Greenshields (1935), is linear in the speed:"}],
      "Text"],
 mathD[greenshieldsEq],

 Cell[TextData[{"so that flow is zero both at empty road (",
   mathI[RowBox[{"\[Rho]", "=0"}]], ") and at bumper-to-bumper jam (",
   mathI[RowBox[{"\[Rho]", "=", rhoMax}]],
   ") and is maximised at the critical density ",
   mathI[RowBox[{rhoCrit, "=", RowBox[{rhoMax, "/", "2"}]}]],
   ". Substituting into the conservation law makes it formally a \
",
   StyleBox["Burgers-type", FontSlant -> Italic],
   " equation \[LongDash] a scalar conservation law with a nonlinear \
flux. Characteristic speeds are ",
   mathI[RowBox[{"\[PartialD]", "Q", "/", "\[PartialD]", "\[Rho]"}]],
   "; they're positive (downstream) at low density and ",
   StyleBox["negative", FontSlant -> Italic],
   " (upstream) at high density. A shock forms the moment two \
characteristics collide, and the shock's own speed is ",
   mathI[RowBox[{"(",
     RowBox[{SubscriptBox["Q", "L"], "-", SubscriptBox["Q", "R"]}],
     ")", "/", "(",
     RowBox[{SubscriptBox["\[Rho]", "L"], "-", SubscriptBox["\[Rho]", "R"]}], ")"}]],
   " (the Rankine\[Dash]Hugoniot condition from gas dynamics)."}],
      "Text"],

 Cell[TextData[{"We discretise with the Godunov upwind flux, the \
textbook choice for Burgers-type equations. A CFL condition ",
   mathI[RowBox[{"\[CapitalDelta]t", "\[LessEqual]",
     RowBox[{"\[CapitalDelta]x", "/", vMaxSym}]}]],
   " keeps the scheme stable:"}], "Text"],

 inputCell["lwrRes = LWR`SimulateLWR[];
LWR`PlotLWR[lwrRes]"],
 outputCell[lwrPlot],

 Cell[TextData[{"The animated version makes the shock/rarefaction \
asymmetry obvious. The initial high-density block has a sharp \
upstream edge (left) and a sharp downstream edge (right); the two \
evolve very differently:\n",
   "\[Bullet] Upstream, density rises from 20 to 120 veh/km and \
characteristic speed drops to negative values. Characteristics \
collide: a ",
   StyleBox["shockwave", FontSlant -> Italic],
   " forms and propagates backward \[LongDash] this is the classic \
signature of a jam viewed from the roadside.\n",
   "\[Bullet] Downstream, density drops from 120 to 20 veh/km and \
characteristics fan outward. No shock forms; instead the block \
melts into a smooth ",
   StyleBox["rarefaction wave", FontSlant -> Italic],
   " as cars accelerate out of the jam.\n",
   "Notice that the shock moves ",
   StyleBox["backward", FontSlant -> Italic],
   " even though every individual car moves forward \[LongDash] the \
very same phenomenon we'll see in the microscopic models later. The \
Greenshields fundamental diagram is a crude fit to empirical data, \
but LWR already captures the qualitative structure of a real jam."}],
      "Text"],
 inputCell["lwrFrames = Table[CoreAnimations`LWRCorridorFrame[lwrRes, k],
  {k, 1, Length[lwrRes[\"t\"]], 2}];
ListAnimate[lwrFrames, AnimationRate -> 4]"],
 outputCell[lwrAnim],

 Cell["3. Adding a second equation: Payne-Whitham", "Section"],
 Cell[TextData[{"LWR has an obvious shortcoming: velocity is \
always the equilibrium value ",
   mathI[RowBox[{"V", "(", "\[Rho]", ")"}]],
   " on the fundamental diagram, which implicitly assumes drivers \
adapt infinitely fast. Real drivers take a second or two to react, \
and they anticipate the state of traffic downstream. Payne (1971) \
and Whitham (1974) independently added a momentum equation to \
capture both effects: velocity relaxes towards equilibrium with \
time constant ", mathI[tauSym],
   " (reaction) and is pushed by a pressure-like anticipation term ",
   mathI[SuperscriptBox[c0Sym, "2"]], ":"}], "Text"],
 mathD[payneEq],

 Cell[TextData[{"The system is now second-order hyperbolic: two \
characteristic speeds, ", mathI[RowBox[{"v", "\[PlusMinus]", c0Sym}]],
   ", generalising the single LWR characteristic ",
   mathI[RowBox[{"V'", "(", "\[Rho]", ")"}]],
   ". With the extra degree of freedom the density can be \
temporarily out of equilibrium, which is exactly what produces the \
stop-and-go oscillations seen on real motorways near merges."}],
      "Text"],

 inputCell["pwRes = PayneWhitham`SimulatePW[];
pwFrames = Table[CoreAnimations`PWCorridorFrame[pwRes, k],
  {k, 1, Length[pwRes[\"t\"]], 2}];
ListAnimate[pwFrames, AnimationRate -> 4]"],
 outputCell[pwAnim],

 Cell[TextData[{"Payne\[Dash]Whitham has a well-known pathology \
\[LongDash] Daganzo's (1995) \"Requiem for second-order models\" \
showed it can produce information travelling backward faster than \
any car (cars effectively react to information from behind). Modern \
second-order models (ARZ, GKT) fix this by making the pressure \
anisotropic: drivers respond only to what's in front. We keep PW \
here because it's the textbook second-order PDE and the fix is an \
exercise in swapping the sign on one spatial derivative \[LongDash] \
the qualitative stop-and-go story is already visible."}], "Text"],

 Cell["4. The microscopic view: Intelligent Driver Model", "Section"],
 Cell[TextData[{"Zooming all the way in to an individual vehicle, \
the Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000) \
is a second-order ODE for the longitudinal dynamics:"}], "Text"],
 mathD[idmAccEq],
 Cell[TextData[{"where ", mathI[vAlpha], " and ", mathI[sAlpha],
   " are the speed and gap to the vehicle in front, ", mathI[dvAlpha],
   " is the approach rate, and the desired gap is"}], "Text"],
 mathD[idmGapEq],

 Cell[TextData[{"Each parameter has a direct physical meaning that \
a traffic engineer can calibrate to local data: ",
   mathI[v0Sym], " is the desired free-flow speed, ",
   mathI[s0Sym], " the minimum bumper gap at standstill (for real \
drivers ~2 m), ",
   mathI[THead],
   " the safe time headway (UK typical 1.5\[Dash]2 s), ",
   mathI["a"], " and ", mathI["b"],
   " the comfortable acceleration and braking. Crucially, the \
velocity-difference term ",
   mathI[RowBox[{"v", "\[CenterDot]", "\[CapitalDelta]v", "/(",
     RowBox[{"2", SqrtBox[RowBox[{"a", "\[CenterDot]", "b"}]]}], ")"}]],
   " in ", mathI[sStar],
   " makes IDM collision-free \[LongDash] unlike its ancestor Bando \
OVM, IDM responds to the ",
   StyleBox["rate of approach", FontSlant -> Italic],
   " rather than just the current gap, so a fast car approaching a \
slower one in front starts braking in time. This is the same \
principle car-makers build into modern adaptive cruise control."}],
      "Text"],
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

 Cell[TextData[{"Linear-stability analysis of the homogeneous \
solution yields a string-stability criterion: uniform flow is \
linearly unstable when the slope of the fundamental diagram exceeds \
a threshold involving ",
   mathI["T"], ", ", mathI["a"], " and ", mathI["b"],
   ". The pre-rendered run sits comfortably on the unstable side of \
that criterion, which is why the small ",
   mathI[RowBox[{"v", "\[Rule]", "v", "/", "2"}]],
   " kick does not decay. Empirically, stop-and-go jam fronts on \
real motorways propagate upstream at 5\[Dash]7 m/s essentially \
independent of the inflow \[LongDash] IDM reproduces this number \
from first principles with no further tuning."}], "Text"],

 Cell[TextData[{"Sugiyama et al. (2008) reproduced exactly this \
experiment with real drivers on a real circular track. The jam \
formed spontaneously, propagated backward, and persisted as long as \
the experiment ran. Their video is highly recommended alongside \
the simulation: ",
   ButtonBox["youtube.com/watch?v=Rryu85BtALM",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://www.youtube.com/watch?v=Rryu85BtALM"],
                    None}],
   ". The match with our run is qualitative rather than bit-exact, \
but the same jam-formation mechanism, the same backward \
propagation, and the same insensitivity to the specific \
perturbation all show up in the real-world data."}], "Text"],

 Cell["5. The simplest car-following that jams: Bando OVM", "Section"],
 Cell[TextData[{"IDM has seven parameters and a velocity-difference \
term. Bando et al. (1995) asked how much of this is necessary to \
get phantom jams and answered: none of it. Their Optimal Velocity \
Model relaxes speed towards a target velocity that depends ",
   StyleBox["only", FontSlant -> Italic],
   " on the current gap to the vehicle ahead:"}], "Text"],
 mathD[bandoEq],
 Cell[TextData[{"with the standard sigmoidal optimal-velocity \
function"}], "Text"],
 mathD[bandoVopt],
 Cell[TextData[{"This is three parameters total \[LongDash] ",
   mathI[kappa], " (sensitivity), ", mathI[vMaxOVM],
   " (free-flow speed), ", mathI[scSym],
   " (crossover gap) \[LongDash] and no reaction to the rate of \
approach. Linear-stability analysis of the uniform-flow fixed point \
gives a clean criterion: the solution is stable exactly when ",
   mathI[RowBox[{"2", SubscriptBox["V", "opt"], "'(", "s*", ")",
     "\[LessEqual]", kappa}]], ". Cross that line and a ",
   StyleBox["Hopf bifurcation", FontSlant -> Italic],
   " kicks in \[LongDash] the uniform solution loses stability and \
the system collapses onto a limit cycle, which on the ring appears \
as a rotating stop-and-go wave. This is arguably the cleanest \
mathematical explanation for why phantom jams exist at all: they \
are the attractor of a dynamical bifurcation."}], "Text"],

 Cell[TextData[{"Bando OVM does have a real weakness: with no \
velocity-difference term, a fast car can drive straight into a \
slow one if the parameters are wrong. The successor Full Velocity \
Difference model (Jiang et al. 2001) adds exactly that term, which \
is effectively IDM's approach-rate piece. IDM, OVM, and FVD form a \
family: increasing physical realism, increasing parameter count, \
same emergent behaviour. For ",
   mathI[RowBox[{kappa, "=0.4"}]],
   " on an 800 m ring of 18 vehicles we sit firmly below the \
stability threshold, and the Hopf instability grows over the 2000 \
s run:"}], "Text"],

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
 Cell[TextData[{"IDM and Bando are ODEs; a lot can be simplified by \
going fully discrete. Nagel and Schreckenberg (1992) proposed the \
minimal CA that still reproduces realistic traffic dynamics. Space \
is discretised into 7.5 m cells (one car length), time into 1 s \
steps, velocity into an integer in [0, ",
   mathI[vMaxCell], "]. At each step every vehicle applies four rules:"}],
      "Text"],
 mathD[naschRule1], mathD[naschRule2], mathD[naschRule3], mathD[naschRule4],

 Cell[TextData[{"where ", mathI["d"],
   " is the number of empty cells to the next vehicle. Rule 3 is \
the crucial ingredient \[LongDash] without it (set ",
   mathI[RowBox[{pSlow, "=0"}]], "), NaSch reduces to the ",
   StyleBox["asymmetric simple exclusion process (ASEP)", FontSlant -> Italic],
   " and the steady state is boring: either uniformly dense or \
uniformly sparse. The random slowdown breaks that degeneracy and \
spontaneously nucleates jams. NaSch is the cellular cousin of \
IDM/OVM: same stop-and-go physics, same backward-propagating waves, \
from four lines of integer arithmetic per car per step. It scales \
to millions of vehicles easily, which is why real urban \
microsimulators (MATSim, SUMO) use CA or CA-like rules internally."}],
      "Text"],

 inputCell["nsRes = NagelSchreckenberg`SimulateNaSch[
  \"T\" -> 300, \"nVehicles\" -> 80, \"roadLength\" -> 300];
nsFrames = Table[
  CoreAnimations`NaSchHighwayFrame[nsRes, t],
  {t, 1, nsRes[\"T\"], 5}];
ListAnimate[nsFrames, AnimationRate -> 2]"],
 outputCell[nsAnim],

 Cell[TextData[{"The backward-propagating jam waves appearing in \
the space-time diagram are the same bands we've already seen in the \
IDM animation (Section 4) and hinted at in LWR (Section 2). Three \
completely different mathematical frameworks \[LongDash] integer CA, \
nonlinear ODE system, hyperbolic PDE \[LongDash] converge on the \
same qualitative behaviour. This is a hallmark of universality: \
the stop-and-go wave is an emergent feature of the driver-follower \
relationship, robust to the choice of formalism."}], "Text"],

 Cell[TextData[{"Extending to two lanes with a symmetric lane-change \
rule (incentive: would have to brake in current lane; safety: \
sufficient gap ahead and behind in the target lane; target cell \
free) recovers the classic multi-lane redistribution:"}], "Text"],
 inputCell["ns2Res = NagelSchreckenberg2Lane`SimulateNaSch2[
  \"T\" -> 200, \"nVehicles\" -> 150, \"roadLength\" -> 500];
NagelSchreckenberg2Lane`PlotNaSch2[ns2Res]"],
 outputCell[ns2Plot],

 Cell["7. Intersections as queues: M/D/1", "Section"],
 Cell[TextData[{"Continuous flow on a link is only half the story; \
intersections are where vehicles accumulate, and a signalised \
junction looks exactly like a server that alternates between \
available (green) and unavailable (red). With Poisson arrivals at \
rate ", mathI[lambda],
   " and deterministic service during green at rate ", mathI[mu],
   ", the Pollaczek\[Dash]Khinchine formula for the M/D/1 queue gives"}],
   "Text"],
 mathD[md1LqEq], mathD[md1WqEq],
 Cell[TextData[{"both diverging as utilisation ", mathI[rhoDef],
   " approaches 1. Deterministic service is optimistic \[LongDash] \
the more general M/G/1 formula scales these by a factor ",
   mathI[RowBox[{RowBox[{"(", "1", "+", SuperscriptBox["C", "2"], ")"}],
     "/", "2"}]],
   " where ", mathI["C"],
   " is the coefficient of variation of the service time. For \
deterministic service ", mathI[RowBox[{"C", "=0"}]],
   ", halving the result; for exponential (M/M/1) ", mathI[RowBox[{"C", "=1"}]],
   " doubles it. So M/D/1 is the lower bound \[LongDash] real \
intersections are worse."}], "Text"],

 Cell[TextData[{"Traffic engineers read the practical threshold ",
   mathI[RowBox[{"\[Rho]", "=0.85"}]],
   " as the onset of severe congestion. On a tandem of ",
   mathI["n"],
   " signalised intersections along a corridor we approximate total \
delay by summing per-intersection delays \[LongDash] a \
Jackson-network-style approximation that ignores \
platoon-preservation between signals but is accurate enough for \
first-pass planning. It's exactly wrong in one important case: when \
a queue at one junction backs up past the stop-line of the previous \
one (",
   StyleBox["spillback", FontSlant -> Italic],
   "), the M/D/1 formula no longer applies because blocking breaks \
the Markov property. That's the regime where you have to switch to \
microsimulation."}], "Text"],

 inputCell["qRes = Queueing`SimulateQueueing[];
Queueing`PlotQueueing[qRes]"],
 outputCell[qPlot],

 Cell["8. Optimising the signal timing", "Section"],
 Cell[TextData[{"The queueing formulas above give us an objective \
function we can optimise. Given a corridor of signals at which \
arrivals vary by intersection, what green times minimise total \
delay subject to a shared green-time budget? Each intersection's \
delay combines the M/D/1 waiting term ",
   mathI[RowBox[{WqSym, "(", "\[Rho]", ")"}]],
   " above with Webster's (1958) uniform-delay term for the fraction \
of the cycle a typical vehicle spends waiting at red:"}], "Text"],
 mathD[websterEq],
 Cell[TextData[{"where ", mathI[Cycle], " is cycle length and ",
   mathI[greenG],
   " is green time at the intersection. Summing these over \
intersections with a linear total-green constraint gives a smooth, \
strictly convex problem that ",
   StyleBox["FindMinimum[\[CenterEllipsis], Method -> \"InteriorPoint\"]",
     "Input"],
   " solves in milliseconds."}], "Text"],

 Cell[TextData[{"Below is a five-intersection Union Street-like \
corridor with asymmetric demand (arrival rates 0.08, 0.15, 0.22, \
0.17, 0.06 veh/s). A uniform 50 s allocation to every intersection \
gives 8.4 veh\[CenterDot]s of total delay; the optimal allocation \
(which pushes green time from the two quiet junctions to the \
central busy one) cuts that to 7.7 \[LongDash] an 8.4% improvement \
at zero capital cost. This is also the cleanest argument for why \
urban signal plans should be data-driven: you get a double-digit \
percentage delay reduction just by solving a convex program with \
local counts. Beyond this, coordinated ",
   StyleBox["green waves", FontSlant -> Italic],
   " (SCOOT, MOVA) optimise offsets between intersections to let \
platoons through without stopping, adding another large chunk of \
improvement."}], "Text"],

 inputCell["soRes = SignalOptimisation`SimulateSignalOpt[];
SignalOptimisation`PlotSignalOpt[soRes]"],
 outputCell[soPlot],
 Cell[TextData[{"The signal controller itself \[LongDash] \
UK-standard 90 s cycle, 4 s amber intergreen, round-robin approach \
phases \[LongDash] is a few lines of data-driven code:"}], "Text"],
 inputCell["icRes = IntersectionControl`SimulateSignalDemo[];
IntersectionControl`PlotSignalDemo[icRes]"],
 outputCell[icPlot],

 Cell["9. Network-level assignment: Wardrop equilibrium", "Section"],
 Cell[TextData[{"Zooming all the way back out: at the scale of a \
whole city, the question is no longer \"what does one link look \
like?\" but \"how do drivers distribute themselves across the \
network?\". Wardrop's first principle (1952) gives the answer: at \
equilibrium, all used paths between the same origin-destination \
pair have equal cost. No driver can unilaterally improve by \
switching routes. Formally, for all paths ",
   mathI["p, q"], " used by the same OD pair:"}], "Text"],
 mathD[wardropEq],
 Cell[TextData[{"Beckmann (1956) showed this is equivalent to the \
unique minimiser of the convex program"}], "Text"],
 mathD[beckmannEq],
 Cell[TextData[{"with the empirical Bureau of Public Roads link-cost \
function"}], "Text"],
 mathD[bprEq],
 Cell[TextData[{"where ", mathI[te0],
   " is the free-flow travel time, ", mathI[CeSym],
   " the capacity, and the tuning parameters are the canonical ",
   mathI[RowBox[{alphaSym, "=0.15"}]], ", ",
   mathI[RowBox[{betaSym, "=4"}]],
   ". Beckmann's integral is smooth and strictly convex, so ",
   StyleBox["FindMinimum", "Input"],
   " with an interior-point method converges in a handful of \
iterations. It's worth noting the economics here: Wardrop's \
user-equilibrium is ",
   StyleBox["not", FontSlant -> Italic],
   " the system optimum \[LongDash] the selfish route choice is in \
general suboptimal in the total-travel-time sense. The ratio of the \
two, the ",
   StyleBox["price of anarchy", FontSlant -> Italic],
   ", is bounded above by 4/3 for BPR-style costs (Roughgarden 2002). \
A striking corollary is ",
   StyleBox["Braess's paradox", FontSlant -> Italic],
   ": adding a road to a network can make ",
   StyleBox["everyone", FontSlant -> Italic],
   " slower. The constrained Aberdeen topology is exactly the kind \
of network where Braess-like effects show up."}], "Text"],

 inputCell["naRes = NetworkAssignment`SimulateAssignment[];
NetworkAssignment`PlotAssignment[naRes]"],
 outputCell[naPlot],
 Cell[TextData[{"On the curated 15-node Aberdeen network (Haudagain, \
Bridge of Don, Bridge of Dee, Union Street, Anderson Drive, AWPR \
junction, etc.) with 4 OD pairs summing to 8500 veh/h, the solver \
produces saturation levels that track real-world experience:"}],
      "Text"],
 inputCell["abRes = AberdeenNetwork`SimulateAberdeenNetwork[];
AberdeenNetwork`PlotAberdeenNetwork[abRes]"],
 outputCell[abPlot],
 Cell[TextData[{"Union Street (saturation 3.0\[Times] capacity) and \
Anderson Drive south (2.4\[Times]) emerge as the dominant \
bottlenecks, exactly the corridors Aberdonians complain about. The \
model has no knowledge of these being the busy corridors \
\[LongDash] they fall out of the Wardrop equilibrium purely from the \
network topology, link capacities, and OD demand."}], "Text"],

 Cell["10. Time-varying demand", "Section"],
 Cell[TextData[{"Real traffic isn't a static snapshot: rush-hour \
demand builds up over 30\[Dash]60 minutes, peaks, and decays. A \
quick way to capture this without switching to a full dynamic \
traffic assignment (DTA) model is to solve Wardrop period by \
period, with some fraction of each period's flow carrying over as \
residual congestion that reduces the effective capacity in the \
next period:"}], "Text"],

 inputCell["daRes = DynamicAssignment`SimulateDynamicAssignment[];
DynamicAssignment`PlotDynamicAssignment[daRes]"],
 outputCell[daPlot],
 Cell[TextData[{"Three qualitative features of real rush hours fall \
out naturally: (a) flow on the ",
   StyleBox["shortest route", FontSlant -> Italic],
   " saturates first; (b) flow then ",
   StyleBox["spills over", FontSlant -> Italic],
   " onto secondary routes as the fastest route's BPR cost climbs; \
and (c) the link-cost heatmap decays ",
   StyleBox["asymmetrically", FontSlant -> Italic],
   " after the demand peak, because residual congestion from the \
peak persists into the post-peak period even as new demand is \
falling. This hysteresis is a well-known empirical fact on real \
motorways (the afternoon rush clears slower than the morning rush \
arrived), and it comes out of the coupling between demand and \
capacity without any explicit dynamical ingredients."}], "Text"],

 Cell["11. Stitching it all together: OSM-based Aberdeen", "Section"],
 Cell[TextData[{"The repo ships two larger simulations that combine \
ingredients from almost every previous section. They're omitted \
from the attached bundle because they need OpenStreetMap JSON \
caches (1\[Dash]9 MB), but the pre-rendered animations are below \
for reference; see ",
   ButtonBox["github.com/mthiel74/TrafficJAms",
     BaseStyle -> "Hyperlink",
     ButtonData -> {URL["https://github.com/mthiel74/TrafficJAms"],
                    None}],
   " for the code."}], "Text"],

 Cell[TextData[{"The ",
   StyleBox["city-centre simulation", FontWeight -> Bold],
   " (",
   StyleBox["AberdeenCityNetwork.wl", "Input"],
   ") parses a 1.3 MB Overpass extract around Marischal College into \
a 1397-node / 2773-edge drivable graph, then runs 150 IDM agents \
routed by ", StyleBox["FindShortestPath", "Input"],
   " with on-arrival re-routing. It takes ~1 s wall time and \
reproduces the morning-rush pattern around Union Street pretty \
well:"}], "Text"],
 outputCell[loadAnim["aberdeen_city.gif"]],

 Cell[TextData[{"The ",
   StyleBox["full-city v2 simulation", FontWeight -> Bold],
   " (",
   StyleBox["AberdeenFullCity.wl", "Input"],
   ") scales this up to the whole council area \[LongDash] an \
8.9 MB extract with 11,925 drivable nodes after \
strongly-connected-component restriction. On top of IDM car-\
following it adds 151 OSM-tagged signalised junctions (the signal \
controller from Section 8), ~1000 roundabout-tagged edges with \
TRL\[ThinSpace]281 gap-acceptance yield, a hierarchy-weighted \
routing (motorway factor 0.5, residential 1.2), staggered spawn, \
and a 10% probabilistic detour at each junction. Six hundred \
agents, six minutes of simulated time, ~30 s wall time:"}], "Text"],
 outputCell[loadAnim["aberdeen_full.gif"]],

 Cell[TextData[{"The Aberdeen Western Peripheral Route \[LongDash] \
opened 2018 to divert through-traffic around the city \[LongDash] \
is the big ring visible on the west/north perimeter, and you can \
watch it carry the lion's share of through-flow exactly as intended. \
This is the pay-off of the whole tour: once you have microscopic \
car-following, intersection control, and a real street graph, the \
city-scale pattern emerges without further tuning."}], "Text"],

 Cell["12. Conclusions", "Section"],
 Cell[TextData[{"A tour through five orders of magnitude of spatial \
scale \[LongDash] from a single vehicle's ODE up to a \
11\[ThinSpace]925-node city-wide multi-agent simulation \
\[LongDash] and a few recurring themes:"}], "Text"],

 Cell[TextData[{
   "\[Bullet] ",
   StyleBox["Phantom jams are universal.", FontWeight -> Bold],
   " The same backward-propagating stop-and-go wave appears in a \
hyperbolic PDE (LWR shock), a 7-parameter nonlinear ODE system \
(IDM), a 3-parameter ODE system (Bando OVM), and an integer \
cellular automaton (NaSch). Four completely different formalisms, \
one qualitative answer. The wave itself is a genuine emergent \
feature of the follow-the-leader relationship, not an artefact of \
any specific model.\n",
   "\[Bullet] ",
   StyleBox["Stability is a bifurcation.", FontWeight -> Bold],
   " For every model the transition from free flow to stop-and-go \
is a well-defined instability \[LongDash] a Hopf bifurcation in \
Bando OVM, a string-stability criterion in IDM, a phase transition \
in NaSch. Crossing the threshold doesn't just degrade the solution, \
it changes its qualitative character.\n",
   "\[Bullet] ",
   StyleBox["Selfish routing is suboptimal.", FontWeight -> Bold],
   " The Wardrop equilibrium is a Nash equilibrium in the \
route-choice game, and like most Nash equilibria it's not socially \
optimal. Braess's paradox is the extreme version: adding capacity \
can make everyone worse off. This is why real-world network \
interventions (bus gates, road pricing, the AWPR) are political \
hard decisions even when the traffic physics is clear.\n",
   "\[Bullet] ",
   StyleBox["Signal timing is convex.", FontWeight -> Bold],
   " The objective combining Webster's uniform delay and the M/D/1 \
waiting term is smooth and strictly convex \[LongDash] a \
tractable optimisation that routinely saves ~10% delay. Networked \
control (green waves) and adaptive systems like SCOOT/MOVA push \
that further but build on the same convex foundation.\n",
   "\[Bullet] ",
   StyleBox["Wolfram is well-suited to this.", FontWeight -> Bold],
   " The entire port runs in ~40 seconds end-to-end on a laptop, \
with no external dependencies beyond ",
   StyleBox["FindMinimum", "Input"], ", ",
   StyleBox["FindShortestPath", "Input"], ", ",
   StyleBox["Graph", "Input"], ", and ",
   StyleBox["NDSolveValue", "Input"],
   ". Wolfram's built-in OSM and geocomputation support \
(",
   StyleBox["OpenStreetMapData", "Input"], ", ",
   StyleBox["Import[\"*.osm\"]", "Input"],
   ") makes it plausible to skip the Python/NetworkX/OSMnx stack \
for research-scale work; the GitHub repo has both ports side-by-\
side."}], "Text"],

 Cell["13. Implementation notes", "Section"],
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

 Cell["14. References", "Section"],
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
