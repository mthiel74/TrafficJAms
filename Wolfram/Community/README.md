# Wolfram Community post

This directory bundles everything needed to post the project as an
interactive notebook on
[community.wolfram.com](https://community.wolfram.com/).

## What to upload

| File | Size | Role |
|---|---:|---|
| `trafficjams.nb` | 22 MB | Main notebook. Math + Input cell (code) + Output cell (pre-rendered figure or animated GIF) per section. |
| `trafficjams-packages.zip` | 30 KB | The 13 `.wl` packages the Input cells load. |

**That's it — no GIF or MP4 attachments needed.**

The 7 animations are embedded directly inside `trafficjams.nb` as
`AnimatedImage[...]` objects. The frames are imported from the GIFs
at build time and serialized into the notebook file, so the animations
travel with the `.nb` itself.

## Why the animations now actually play

Wolfram Community's web viewer **pre-renders every notebook to static
HTML on the server**. Anything that needs a live kernel
(`ListAnimate`, `Animate`, `Manipulate`, `Dynamic`) collapses to a
still image of its first frame. The one exception is `AnimatedImage`
constructed from a GIF — the pre-renderer passes the GIF payload
through as an `<img>`, and the browser loops it natively. We build
every animation in the notebook with `AnimatedImage`, so playback
works in the browser without anything extra.

If a reader downloads the notebook and opens it locally, the
`AnimatedImage` plays there too (same mechanism as any other imported
GIF).

## What the reader does

1. Download `trafficjams.nb` and `trafficjams-packages.zip` from the
   post.
2. Put them in the same directory.
3. Unzip `trafficjams-packages.zip` in that directory. This drops the
   13 `.wl` files next to the notebook:

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

4. Open the notebook (either in the Community browser viewer, in
   Wolfram Desktop, or in Wolfram Cloud). All figures and animations
   are already there, so reading it needs no evaluation.
5. To regenerate a figure, evaluate its Input cell. The first cell
   in "Getting started" sets `SetDirectory[NotebookDirectory[]]` and
   `Get`s every package.

## OSM-based Aberdeen simulations

Two extra simulations (`AberdeenCityNetwork.wl`, `AberdeenFullCity.wl`)
need OSM JSON caches (1.3 MB / 8.8 MB) and aren't bundled here. The
relevant section in the notebook embeds their pre-rendered GIFs for
reference and points at
[github.com/mthiel74/TrafficJAms](https://github.com/mthiel74/TrafficJAms)
for the source + caches.

## Rebuilding

```bash
wolframscript -file build_notebook.wls
```

regenerates `trafficjams.nb`, pulling GIFs and PNGs from `../results/`
and wrapping them in `AnimatedImage[...]` / `Image[...]`.

```bash
zip -9 trafficjams-packages.zip *.wl
```

regenerates the package archive.
