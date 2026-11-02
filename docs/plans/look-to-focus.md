# Look To Focus

Status: design, not built. Proposed 2026-11-02. Nothing in this plan ships
until phase 0's measurement says webcam gaze is good enough; Kit decides
whether it continues past that.

Goal: dictate into the text field you're looking at. Look at a field, press
the dictation shortcut, and Hex focuses that field before it pastes, with no
mouse or click. Hex already types (paste into the focused field); this adds
choosing the field. It's a Labs feature, off by default, on a webcam Hex
reads itself, with no eye-tracking hardware, vendor SDK, or subscription.

## What It Does

```ts
Labs › Look to focus on, calibrated
└── look at a text field, press the dictation shortcut
    ├── camera opens (or is already warm) -> gaze samples at ~30 Hz
    ├── first stable fixation after the press (≥ 150 ms within 2°)
    │   └── hit-test editable elements near it, within the snap radius
    │       ├── one editable element wins -> focus it (caret moves, visibly)
    │       │   while the dictation keeps recording
    │       └── none, or ambiguous -> change nothing; paste goes where focus already is
    ├── release -> transcribe -> paste into the focused field (unchanged path)
    └── camera closes once the press is over (unless kept warm)
```

- The dictation shortcut is the click. There is no dwell click, no gaze
  gesture, and the pointer doesn't move unless the element refuses
  Accessibility focus (see [Focusing](#focusing)).
- Gaze only chooses where the paste goes. It never starts, ends, delays, or
  discards a capture, and a gaze failure (no face, camera busy, model
  missing) leaves dictation exactly as it is today.
- Without calibration the feature does nothing but say it needs calibrating.

Later phases may add a gaze marker on the HUD and continuous pointer
control; neither is in scope until the snap-to-field path is proven.

## Why A Webcam, And Why Snapping

Research (2026-11-02): consumer Tobii trackers have a usable SDK only on
Windows (Stream Engine, access by request; the only Rust binding,
`tobii-sys`, is from 2018), Tobii Pro's cross-platform C SDK needs research
hardware, and Eyeware Beam's SDK needs its app running and a subscription.
Webcam gaze needs nothing the person doesn't have.

Webcam gaze is accurate to about 2–3°, roughly 2–5 cm on screen at 60 cm.
That can't point at a character, but it can tell which of a few text fields
you mean, so the design snaps to editable elements Hex already finds through
Accessibility rather than trusting the raw point.

## Pipeline

```ts
gaze worker thread (utility QoS, bounded; never on the audio or UI thread)
├── camera        latest frame only, ~30 Hz, 640×480; older frames dropped
├── face          BlazeFace short-range -> face box (Apache-2.0, ~0.2 MB)
├── landmarks     Face Mesh V2 on the crop -> 478 points incl. 10 iris (Apache-2.0, ~3 MB)
├── features      iris centre relative to each eye's corners and lids,
│                 head pose from a fixed 3D face model (PnP), eye openness
├── calibration   per-person ridge regression features -> screen point,
│                 fitted from the calibration dots
├── filter        One Euro filter; fixation = ≥ 150 ms inside 2°
└── target        Accessibility hit-test around the fixation -> one editable element
```

- Both models are Google's MediaPipe weights converted to ONNX (community
  conversions exist; we convert and pin our own and check outputs against
  the TFLite reference). They're downloaded on first use through `net`,
  SHA-256 pinned, beside the speech models, credited like other model
  licences. A missing model downloads in the background and the feature
  waits, as `transcription/speech_detector` does.
- No model trained on gaze datasets ships. Appearance-based gaze models
  (L2CS-Net, UniGaze) are more robust to head movement, but their training
  data (Gaze360, MPIIFaceGaze, ETH-XGaze) is reported as research-use only.
  Revisit only with a clear licence.
- Calibration is per person and per display arrangement: a full-screen Rune
  overlay shows 9 dots (then a 5-dot validation pass that reports the error
  in points and estimated cm). It's stored owner-only in
  `<support>/gaze-calibration.json`, keyed by the display arrangement, and
  holds only regression weights and the validation error, never images.

## Focusing

```ts
target(fixation, radius)
├── macOS    AXUIElementCopyElementAtPosition at the fixation and a ring of
│            points within the radius -> nearest element whose role is
│            editable (AXTextField, AXTextArea, AXComboBox, AXSearchField, or
│            AXEditable / a web area's editable node)
├── Windows  IUIAutomation::ElementFromPoint, same ring, ValuePattern/TextPattern editable   (phase 2)
└── Linux    AT-SPI GetAccessibleAtPoint, EDITABLE state                                    (phase 2, X11 first)

focus(element)
├── raise its window, set AXFocused = true (Windows: SetFocus; Linux: grab_focus)
├── verify the focused element is now that element
└── refused -> one synthetic click at the element's centre, pointer returned
    to where it was; still refused -> leave focus alone and log why
```

- Snap radius starts at the calibration's measured error (at least 1.5 cm).
  Two editable candidates both within it and closer than 1 cm apart count as
  ambiguous: nothing changes.
- The hit-test and focus run on the gaze worker with the same bounded
  Accessibility messaging timeout `accessibility.rs` uses, so a hung app
  can't stall anything else.
- Focusing a field never moves an existing caret in that field: focus
  restores the field's own selection. A click (the fallback) does move it;
  that's accepted and logged.

## Camera Lifetime And Privacy

- The camera opens only while the feature is on, and by default only from
  the shortcut press until the target is chosen or the press ends. The
  camera light shows while it's open. "Keep camera ready" (off) holds it
  open while Hex is unlocked, trading the light for no start-up delay;
  phase 0 measures that delay to decide whether the option is needed.
- Frames never leave the gaze worker: no disk, no log, no network, no
  History. Logs keep only timings, the fixation's confidence, the chosen
  element's role, and why a target was refused; never gaze coordinates or
  the element's text.
- macOS asks for Camera only when the Labs row is turned on (the bundle
  already has `NSCameraUsageDescription` for iPhone mirroring; its text
  widens to cover this). Accessibility is already required. Windows needs
  the camera privacy switch; Linux opens V4L2 (the Camera portal on
  Wayland in phase 2).
- The camera closes when the screen locks, the feature turns off, or Hex
  quits.

## Invariants To Add

When built, these join `AGENTS.md`:

- Gaze never changes capture boundaries, the hold threshold, tones, or what
  is transcribed; it can only move focus before a paste. A press whose gaze
  resolves after release changes nothing.
- Camera frames are processed in memory on one bounded worker that keeps
  only the latest frame and are never stored or sent.
- With `labs.look_to_focus` off the listener builds no gaze worker: no
  camera, model load, or download.

## Code Shape

```ts
src/gaze/              // new capability folder
├── camera.rs          macOS AVFoundation capture first (Swift bridge, like apple_speech.swift)
├── landmarks.rs       face crop + Face Mesh on the chosen runtime
├── features.rs        pure: landmarks -> feature vector (tested on fixtures)
├── calibration.rs     pure: fit, predict, validation error, persistence
├── fixation.rs        pure: One Euro filter + fixation detection
├── target.rs          Accessibility hit-test and focus (calls accessibility.rs)
└── worker.rs          lifetime, bounded queue, listener hooks
rune_ui/gaze_calibration.rs  full-screen calibration overlay
```

- One gaze source (the webcam), so no `GazeSource` trait yet. A Tobii or
  Beam adapter would be the second source and introduce it then.
- The listener tells the gaze worker about the press (the same edge the
  start tone uses) through a bounded, non-blocking send; the paste worker
  reads "focus changed" only as a log fact. Nothing waits on gaze.
- Labs: a `labs::Feature::LookToFocus` (key `labs.look_to_focus`, audience
  `Dev` until phase 1 passes, macOS only in `on_this_os` with a
  `platform::DIFFERENCES` reason: pending), its Labs row with Calibrate…,
  and `hex settings` keys. `hex gaze calibrate|status` for the CLI.
- Inference runtime is an open question (below); the choice is made in
  phase 0 by measurement.

## Phases

0. **Spike** (`hex dev gaze`, debug builds only): camera, both models,
   calibration, and validation in a dev window that draws the gaze dot and
   prints error and cost. Go/no-go on the developer's Mac with a built-in
   camera at normal distance:
   - median validation error ≤ 3 cm and 90th percentile ≤ 5 cm,
   - ≥ 15 gaze samples a second at under 15% of one core,
   - camera start to first fixation ≤ 600 ms (else "Keep camera ready"),
   - the same numbers with glasses and in dim light, written down either way.
   Results go in `docs/research/gaze-spike.md`.
1. **macOS Labs (Dev audience)**: press-to-focus, calibration overlay, Labs
   row, CLI, feature map entry in `docs/features/`, snap and focus tests on
   fixtures, and a real-app pass (Safari, Chrome, Notes, Slack, VS Code).
2. **Windows and Linux X11**, then Wayland only where AT-SPI extents are
   trustworthy (they often aren't without a global coordinate space).
3. **Everyone audience** once real use says it helps; dedicated trackers
   only if someone needs more accuracy than a webcam gives.

## Open Questions

- Runtime: `tract` (pure Rust, no native library, CPU only) vs `ort`
  (ONNX Runtime, faster with CoreML/DirectML, but a large native library
  in every build). Start with `tract`; switch only if phase 0 misses the
  cost budget.
- Does focus-by-Accessibility work in Electron and web text fields often
  enough, or will the click fallback be the common path?
- Should the HUD show where Hex thinks you're looking during the hold, so a
  wrong field is obvious before the paste?
- Multiple displays: one calibration across all, or one per display?
- How often calibration drifts with posture, and whether a quick 1-dot
  recheck on press failures is enough.
