# Wolfram Community post

`trafficjams.nb` is a ready-to-upload Wolfram Notebook covering the whole
`TrafficJAms` repository at a level suitable for
[community.wolfram.com](https://community.wolfram.com/). Math is typeset
natively via `FormBox[..., TraditionalForm]` (real fractions, Greek,
subscripts) so equations render in proper math fonts both in the
notebook and in the PDF export.

## Contents

1. Abstract and motivation (why Aberdeen?)
2. LWR conservation PDE + Godunov scheme
3. Payne-Whitham second-order momentum equation
4. IDM (Intelligent Driver Model) — phantom jams on a ring road
5. Bando Optimal Velocity Model — Hopf bifurcation
6. Nagel-Schreckenberg stochastic CA (single- and two-lane)
7. M/D/1 intersection queueing + Pollaczek-Khinchine formula
8. Webster-style signal timing optimisation
9. Wardrop equilibrium and the Beckmann convex program (BPR costs)
10. Dynamic traffic assignment under time-varying demand
11. Full Aberdeen OSM multi-agent case study (600 vehicles, 151 signals)
12. Implementation notes
13. References

## Building

```bash
wolframscript -file build_notebook.wls
```

writes both `trafficjams.nb` (≈3.2 MB) and `trafficjams.pdf` (≈2.8 MB)
into this directory. The script reads the PNGs and first frames of the
GIFs from `../results/` and embeds them in the notebook, so run
`wolframscript -file ../RunAll.wls` first if any of those files are
missing or stale.
