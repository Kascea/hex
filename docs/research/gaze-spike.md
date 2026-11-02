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

Run 1 (2026-11-02): MacBook FaceTime HD camera (it gives 1920×1080 though
640×480 is asked for), normal sitting distance, indoor light. Camera
29 frames/s, face found in every frame, Face Mesh 6.5 ms, gaze thread 19% of
one core. Cost is reported but no longer a target: the camera runs only
around a shortcut press, so accuracy comes first.

The first model also fitted head pose and the face's place in the frame. It
was worse than guessing the screen's centre: posture drifted during the run
(the face sank about 1% of the frame per dot row, yaw grew 0.009 → 0.04),
the fit learned that drift as gaze, and the check dots, shown last, sat
outside it. The iris offsets themselves separate the dot columns and rows
cleanly (±0.06 eye widths, wobble about 0.01). Refitting the same recording
(`calibrate --save`), 9 dots fitted, 5 unseen dots checked:

```ts
model                        per sample, median / p90     leave one dot out (14 dots)
├── guess the centre         7.2 / 7.2 cm
├── first model (10 terms)   8.4 / 10.8 cm                3.0 / 7.2 cm
├── iris only                2.5 / 6.2 cm                 3.0 / 6.6 cm
├── iris + squares (chosen)  2.2 / 5.8 cm                 3.1 / 6.1 cm
├── iris + yaw, pitch        2.8 / 5.8 cm                 3.3 / 7.0 cm
└── per-eye iris (contour)   3.0 / 7.2 cm                 3.4 / 6.7 cm
```

With iris + squares, each check dot's mean prediction (what a fixation
lands on) is off by 2.6 cm median and 3.4 cm p90 (1.1 to 3.8 cm across the 5
dots): inside both targets, on one person and five dots. `calibrate` now
fits iris + squares and judges the targets on that per-dot error. Not yet
known: whether it holds across sessions, distances, glasses, and light, and
how fast drift spoils a calibration in use.

```ts
run    camera            setting               fixation median / p90   gaze/s   core
├── 1  FaceTime HD 1080p  sitting, indoor light  2.6 / 3.4 cm (refit)   29.0     19%
└── 2  (next)
```

## Blocker Found

Upstream Hex doesn't build on a Mac without Xcode 26: `apple_intelligence.swift`
uses FoundationModels' `@Generable` macro, whose plugin ships only with
Xcode 26 (not the Command Line Tools). The gaze modules were compiled and
their 16 unit tests run in a scratch harness that includes them unchanged.
