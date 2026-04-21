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

banner["M/D/1 Queueing"];
Get["Queueing.wl"];
qRes = Queueing`SimulateQueueing[];
Queueing`PlotQueueing[qRes, "results/queueing.png"];

banner["Network Assignment"];
Get["NetworkAssignment.wl"];
naRes = NetworkAssignment`SimulateAssignment[];
NetworkAssignment`PlotAssignment[naRes, "results/network_assignment.png"];

Print["\nAll models done. Outputs in ", Directory[] <> "/results/"];
