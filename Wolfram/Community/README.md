# Wolfram Community post

This directory bundles everything needed to post the project as an
interactive notebook to
[community.wolfram.com](https://community.wolfram.com/).

## What to upload

Two attachments + the notebook:

1. **`trafficjams.nb`** — the main notebook (≈ 22 MB). Every section
   has math, an Input cell with runnable code, and an Output cell
   with the pre-rendered figure or animation, so a reader can browse
   it without evaluating anything.
2. **`trafficjams-packages.zip`** — a 30 KB archive of the 13 `.wl`
   packages the Input cells call into.

## How the reader uses it

1. Download both attachments.
2. Put them in the same directory (anywhere on disk).
3. Unzip `trafficjams-packages.zip` *in that directory*. This drops
   the 13 `.wl` files next to `trafficjams.nb`:

   ```
   trafficjams.nb
   IDM.wl
   Bando.wl
   LWR.wl
   PayneWhitham.wl
   NagelSchreckenberg.wl
   NagelSchreckenberg2Lane.wl
   Queueing.wl
   SignalOptimisation.wl
   IntersectionControl.wl
   NetworkAssignment.wl
   DynamicAssignment.wl
   AberdeenNetwork.wl
   CoreAnimations.wl
   ```

4. Open the notebook. Reading it needs no evaluation — all outputs
   are pre-rendered. To regenerate a figure, evaluate its Input cell;
   the first input cell sets
   `SetDirectory[NotebookDirectory[]]` and loads all packages.

## OSM-based Aberdeen simulations

Two extra simulations (`AberdeenCityNetwork.wl`, `AberdeenFullCity.wl`)
need OSM JSON caches (1.3 MB and 8.8 MB) and aren't bundled here. The
relevant section in the notebook embeds their pre-rendered animations
for reference and points at
[github.com/mthiel74/TrafficJAms](https://github.com/mthiel74/TrafficJAms)
for the source.

## Rebuilding

```bash
wolframscript -file build_notebook.wls
```

regenerates `trafficjams.nb` from the source in `build_notebook.wls`.
It pulls pre-rendered PNGs and GIFs from `../results/`, so run
`wolframscript -file ../RunAll.wls` first if any of those are stale.

```bash
zip -9 trafficjams-packages.zip *.wl
```

regenerates the package archive.
