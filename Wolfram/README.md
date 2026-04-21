# TrafficJAms — Wolfram Language Port

This directory contains a Wolfram Language (Mathematica) port of the traffic-flow
models implemented in the Python `trafficjams/` package. Each `.wl` file is a
standalone package that mirrors one model:

| File                              | Model                                    |
|-----------------------------------|------------------------------------------|
| `IDM.wl`                          | Intelligent Driver Model (microscopic)   |
| `Bando.wl`                        | Optimal Velocity Model (Bando)           |
| `LWR.wl`                          | Lighthill–Whitham–Richards (Godunov)     |
| `PayneWhitham.wl`                 | Payne–Whitham second-order PDE           |
| `NagelSchreckenberg.wl`           | Nagel–Schreckenberg stochastic CA        |
| `Queueing.wl`                     | M/D/1 signalised-corridor queue          |
| `NetworkAssignment.wl`            | Beckmann / BPR Wardrop equilibrium       |
| `RunAll.wls`                      | Runs every model and exports PNGs        |

## Running

All files can be evaluated from a notebook, or run headless with
`wolframscript`:

```
wolframscript -file RunAll.wls
```

Output images are written to `results/`. Each package also defines a
`Simulate<Name>[]` function and a `Plot<Name>[result]` function so individual
models can be explored interactively.

## Notes on the port

* The Wolfram versions are intentionally faithful to the Python reference in
  parameter choices and numerical schemes, so results should match up to
  floating-point noise.
* Where Python loops compute things element-wise we use vectorised primitives
  (`MapThread`, `Mod`, matrix-vector products) — the logic is the same.
* Plots use `ArrayPlot` / `ListDensityPlot` / `ListLinePlot`; they mirror the
  matplotlib figures in `results/` at the top level of the repo.
