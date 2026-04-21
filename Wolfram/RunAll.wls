#!/usr/bin/env wolframscript
(* ::Package:: *)

(* RunAll.wls -- execute every Wolfram traffic model and export PNGs.
   Run with:   wolframscript -file RunAll.wls
   Outputs are written to Wolfram/results/. *)

SetDirectory @ DirectoryName @ $InputFileName;

If[!DirectoryQ["results"], CreateDirectory["results"]];

banner[name_] := Print["\n=== ", name, " ==="];

banner["IDM"];
Get["IDM.wl"];
idmRes = IDM`SimulateIDM[];
IDM`PlotIDM[idmRes, "results/idm.png"];

banner["Bando OVM"];
Get["Bando.wl"];
bandoRes = Bando`SimulateBando[];
Bando`PlotBando[bandoRes, "results/bando.png"];

banner["LWR"];
Get["LWR.wl"];
lwrRes = LWR`SimulateLWR[];
LWR`PlotLWR[lwrRes, "results/lwr.png"];

banner["Payne-Whitham"];
Get["PayneWhitham.wl"];
pwRes = PayneWhitham`SimulatePW[];
PayneWhitham`PlotPW[pwRes, "results/payne_whitham.png"];

banner["Nagel-Schreckenberg"];
Get["NagelSchreckenberg.wl"];
nsRes = NagelSchreckenberg`SimulateNaSch[];
NagelSchreckenberg`PlotNaSch[nsRes, "results/nagel_schreckenberg.png"];

banner["Nagel-Schreckenberg (2-lane)"];
Get["NagelSchreckenberg2Lane.wl"];
ns2Res = NagelSchreckenberg2Lane`SimulateNaSch2[];
NagelSchreckenberg2Lane`PlotNaSch2[ns2Res, "results/nasch_2lane.png"];

banner["M/D/1 Queueing"];
Get["Queueing.wl"];
qRes = Queueing`SimulateQueueing[];
Queueing`PlotQueueing[qRes, "results/queueing.png"];

banner["Signal Optimisation"];
Get["SignalOptimisation.wl"];
soRes = SignalOptimisation`SimulateSignalOpt[];
SignalOptimisation`PlotSignalOpt[soRes, "results/signal_optimisation.png"];

banner["Network Assignment"];
Get["NetworkAssignment.wl"];
naRes = NetworkAssignment`SimulateAssignment[];
NetworkAssignment`PlotAssignment[naRes, "results/network_assignment.png"];

banner["Aberdeen Wardrop Network"];
Get["AberdeenNetwork.wl"];
abNetRes = AberdeenNetwork`SimulateAberdeenNetwork[];
AberdeenNetwork`PlotAberdeenNetwork[abNetRes, "results/aberdeen_network.png"];

banner["Aberdeen City Multi-Agent Simulation"];
Get["AberdeenCityNetwork.wl"];
abCityNet = AberdeenCityNetwork`LoadAberdeenNetwork[];
abCitySim = AberdeenCityNetwork`SimulateAberdeenCity[abCityNet,
   "nVehicles" -> 150, "nFrames" -> 120, "T" -> 300];
AberdeenCityNetwork`PlotAberdeenCity[abCitySim, "results/aberdeen_city.png"];

Print["\nAll models done. Outputs in ", Directory[] <> "/results/"];
