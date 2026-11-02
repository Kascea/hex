//! Look to focus, phase 0 (`docs/plans/look-to-focus.md`): webcam gaze,
//! measured by `hex dev gaze`. Built only with the `gaze` Cargo feature, on
//! macOS, in debug builds; nothing in the app uses it yet.
//!
//! The camera ([`camera`]) hands frames with Apple Vision's face box to
//! Face Mesh ([`landmarks`]); [`features`] turns its points into iris and
//! head-pose numbers, [`calibration`] maps those to a screen point for one
//! person, and [`fixation`] smooths the point and finds where it rests.
//! Frames stay in memory on the thread that reads them.

pub(crate) mod calibration;
pub(crate) mod camera;
pub(crate) mod features;
pub(crate) mod fixation;
pub(crate) mod landmarks;
pub(crate) mod spike;
