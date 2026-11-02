//! The webcam, through `gaze.swift`: AVFoundation frames with Apple Vision's
//! face box. Only the latest frame is kept; nothing is written anywhere.

use std::ffi::{CStr, c_char};
use std::time::Duration;

use color_eyre::eyre::{Result, bail};

unsafe extern "C" {
    fn hex_gaze_camera_authorization() -> i32;
    fn hex_gaze_camera_request_access() -> bool;
    fn hex_gaze_camera_start(error: *mut *mut c_char) -> i32;
    fn hex_gaze_camera_stop();
    fn hex_gaze_camera_name() -> *mut c_char;
    fn hex_gaze_camera_next(
        after: u64,
        timeout_ms: i32,
        out: *mut u8,
        capacity: usize,
        info: *mut f64,
    ) -> i32;
    fn hex_gaze_primary_display(out: *mut f64);
    fn hex_gaze_free_string(string: *mut c_char);
}

/// The largest frame the camera gives: 640×480 BGRA, with headroom for a
/// camera that ignores the preset.
const CAPACITY: usize = 1920 * 1080 * 4;

/// A face box in image pixels, origin top left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// One camera frame, BGRA with no row padding.
pub(crate) struct Frame {
    pub sequence: u64,
    pub width: u32,
    pub height: u32,
    /// Capture time in seconds on the host clock.
    pub time: f64,
    pub face: Option<Rect>,
    pub bgra: Vec<u8>,
}

impl Frame {
    pub(crate) fn new() -> Self {
        Self {
            sequence: 0,
            width: 0,
            height: 0,
            time: 0.0,
            face: None,
            bgra: vec![0; CAPACITY],
        }
    }
}

/// The running camera; dropping it stops the camera.
pub(crate) struct Camera(());

impl Camera {
    /// Asks for Camera access if it hasn't been answered, then starts.
    pub(crate) fn start() -> Result<Self> {
        match unsafe { hex_gaze_camera_authorization() } {
            3 => {}
            0 if unsafe { hex_gaze_camera_request_access() } => {}
            _ => bail!(
                "Camera access is off. Allow it for this app in System Settings › Privacy & Security › Camera."
            ),
        }
        let mut error = std::ptr::null_mut();
        if unsafe { hex_gaze_camera_start(&mut error) } != 0 {
            bail!("{}", take_string(error));
        }
        Ok(Self(()))
    }

    pub(crate) fn name(&self) -> String {
        take_string(unsafe { hex_gaze_camera_name() })
    }

    /// Waits for a frame newer than `frame` and reads it into `frame`;
    /// `false` when none came within `timeout`.
    pub(crate) fn next(&self, frame: &mut Frame, timeout: Duration) -> Result<bool> {
        let mut info = [0.0f64; 9];
        let read = unsafe {
            hex_gaze_camera_next(
                frame.sequence,
                i32::try_from(timeout.as_millis()).unwrap_or(i32::MAX),
                frame.bgra.as_mut_ptr(),
                frame.bgra.len(),
                info.as_mut_ptr(),
            )
        };
        match read {
            1 => {}
            0 => return Ok(false),
            _ => bail!("the camera stopped or gave a frame larger than 1920×1080"),
        }
        frame.sequence = info[0] as u64;
        frame.width = info[1] as u32;
        frame.height = info[2] as u32;
        frame.time = info[3];
        frame.face = (info[4] != 0.0).then(|| Rect {
            x: info[5] as f32,
            y: info[6] as f32,
            width: info[7] as f32,
            height: info[8] as f32,
        });
        Ok(true)
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        unsafe { hex_gaze_camera_stop() }
    }
}

/// The primary display in Rune's global space, and its physical size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Display {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    /// Millimetres, 0 when the display doesn't say.
    pub width_mm: f64,
}

impl Display {
    pub(crate) fn primary() -> Self {
        let mut out = [0.0f64; 6];
        unsafe { hex_gaze_primary_display(out.as_mut_ptr()) };
        Self {
            x: out[0],
            y: out[1],
            width: out[2],
            height: out[3],
            width_mm: out[4],
        }
    }

    /// Centimetres per point, when the display reports its size.
    pub(crate) fn cm_per_point(&self) -> Option<f64> {
        (self.width_mm > 0.0 && self.width > 0.0).then(|| self.width_mm / 10.0 / self.width)
    }
}

fn take_string(pointer: *mut c_char) -> String {
    if pointer.is_null() {
        return String::new();
    }
    let string = unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned();
    unsafe { hex_gaze_free_string(pointer) };
    string
}
