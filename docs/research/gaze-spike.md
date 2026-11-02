# Gaze Spike (Look To Focus, Phase 0)

Measurements for [the plan](../plans/look-to-focus.md)'s phase 0 go/no-go.
Started 2026-11-02 on an Apple Silicon Mac (macOS 26.5).

## Running It

```sh
cargo run --features gaze -- dev gaze probe        # camera + Face Mesh, rate and cost each second
cargo run --features gaze -- dev gaze calibrate    # 9 dots to fit, 5 to check, report, save
cargo run --features gaze -- dev gaze track        # a dot where the saved calibration says you look
```

Debug builds only, behind the off-by-default `gaze` feature. The first run
asks for Camera access (for the terminal app running it) and downloads Face
Mesh (4.9 MB, SHA-256 pinned) into `<support>/models/gaze/`. `calibrate`
saves `<support>/gaze-calibration.json` (weights only, owner-only).

## Inference Runtime (decided)

Face Mesh V2 ONNX (`naklitechie/face-landmarks-onnx`, SHA-256
`f38c3321…ac0c`), one 256×256 input, 30 runs after warm-up, one thread:

```ts
runtime                  median    p90       notes
├── tract-onnx 0.21      50.1 ms   51.5 ms   pure Rust; 20/s at a full core: misses the budget
├── ort rc.12, CPU        6.7 ms    7.3 ms   chosen: 30/s ≈ 20% of a core, 15/s ≈ 10%
└── ort rc.12, CoreML    21.7 ms   22.5 ms   slower than CPU for this small model; 1.5 s load
```

`ort` adds about 20 MB and fetches its library at build time, so it stays
behind the `gaze` feature until the spike passes.

## Face Box

The spike takes the face box from Apple Vision
(`VNDetectFaceRectanglesRequest`) on each frame, in `gaze.swift`, instead of
running BlazeFace. Windows and Linux will need BlazeFace (Apache-2.0) or
tracking from the previous frame's points.

## Accuracy And Cost

Not measured yet: needs a person at the camera running `calibrate`. Record
for each run the camera, distance, lighting, glasses, and the report's
lines:

```ts
run    camera    distance  light  glasses   raw median/p90   smoothed median/p90   gaze/s   core
└── (pending)
```

## Blocker Found

Upstream Hex doesn't build on a Mac without Xcode 26: `apple_intelligence.swift`
uses FoundationModels' `@Generable` macro, whose plugin ships only with
Xcode 26 (not the Command Line Tools). The gaze modules were compiled and
their 16 unit tests run in a scratch harness that includes them unchanged.
