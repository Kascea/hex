//! Face Mesh V2 (Google's MediaPipe weights, Apache-2.0, converted to ONNX)
//! on ONNX Runtime's CPU provider: a square crop around the face, rotated so
//! the eyes are level, in; 478 points in image pixels out.

use std::path::PathBuf;

use color_eyre::eyre::{Result, WrapErr, bail, eyre};
use ort::session::Session;
use ort::value::Tensor;

use super::camera::{Frame, Rect};
use super::features::{LANDMARKS, Point};

/// `naklitechie/face-landmarks-onnx`: `face_landmarks_detector.tflite` from
/// `face_landmarker.task` float16 v1, converted by tf2onnx at opset 17.
const MODEL_URL: &str =
    "https://huggingface.co/naklitechie/face-landmarks-onnx/resolve/main/face_landmarks.onnx";
const MODEL_SHA256: &str = "f38c3321ceffbc9e95103480ad38cc3f52e7e1bde2bcee7cd9355d0b9138ac0c";
const MODEL_BYTES: u64 = 4_920_995;
/// The model's square input.
const SIZE: usize = 256;
/// The crop's side as a multiple of the face box's larger side: Vision's
/// box is tight, and Face Mesh wants the whole head with a margin.
const CROP: f32 = 1.6;
/// Below this tilt one pass is enough; above it a second pass runs level.
const LEVEL: f32 = 0.08;
/// Face presence below this (after the sigmoid) is no face.
const PRESENCE: f32 = 0.5;

/// One face's points.
pub(crate) struct Mesh {
    pub points: Vec<Point>,
}

pub(crate) struct FaceMesh {
    session: Session,
    input: Vec<f32>,
}

/// A crop: its centre in image pixels, its side, and its rotation.
#[derive(Clone, Copy, Debug)]
struct Crop {
    cx: f32,
    cy: f32,
    side: f32,
    angle: f32,
}

impl Crop {
    /// Image pixels for crop pixels `(u, v)` (0 to `SIZE`).
    fn to_image(self, u: f32, v: f32) -> (f32, f32) {
        let scale = self.side / SIZE as f32;
        let (lx, ly) = (
            (u - SIZE as f32 / 2.0) * scale,
            (v - SIZE as f32 / 2.0) * scale,
        );
        let (sin, cos) = self.angle.sin_cos();
        (self.cx + lx * cos - ly * sin, self.cy + lx * sin + ly * cos)
    }
}

impl FaceMesh {
    /// Loads the model, downloading it first if it isn't here yet.
    pub(crate) fn load() -> Result<Self> {
        let path = ensure_model()?;
        let failed = |error: &dyn std::fmt::Display| eyre!("Face Mesh didn't load: {error}");
        let session = Session::builder()
            .map_err(|error| failed(&error))?
            .with_intra_threads(1)
            .map_err(|error| failed(&error))?
            .commit_from_file(&path)
            .map_err(|error| failed(&error))?;
        Ok(Self {
            session,
            input: vec![0.0; SIZE * SIZE * 3],
        })
    }

    /// The face's points, or `None` when the model sees no face in the box.
    pub(crate) fn detect(&mut self, frame: &Frame, face: Rect) -> Result<Option<Mesh>> {
        let mut crop = Crop {
            cx: face.x + face.width / 2.0,
            cy: face.y + face.height / 2.0,
            side: face.width.max(face.height) * CROP,
            angle: 0.0,
        };
        let Some(mut mesh) = self.pass(frame, crop)? else {
            return Ok(None);
        };
        let roll = eye_roll(&mesh.points);
        if roll.abs() > LEVEL {
            crop.angle = roll;
            match self.pass(frame, crop)? {
                Some(level) => mesh = level,
                None => return Ok(None),
            }
        }
        Ok(Some(mesh))
    }

    fn pass(&mut self, frame: &Frame, crop: Crop) -> Result<Option<Mesh>> {
        sample(frame, crop, &mut self.input);
        let tensor = Tensor::from_array(([1usize, SIZE, SIZE, 3], self.input.clone()))?;
        let outputs = self.session.run(ort::inputs!["input_12" => tensor])?;
        let (_, points) = outputs["Identity"].try_extract_tensor::<f32>()?;
        let (_, logit) = outputs["Identity_1"].try_extract_tensor::<f32>()?;
        let presence = 1.0 / (1.0 + (-logit[0]).exp());
        if presence < PRESENCE || points.len() < LANDMARKS * 3 {
            return Ok(None);
        }
        let scale = crop.side / SIZE as f32;
        let points = points
            .as_chunks::<3>()
            .0
            .iter()
            .take(LANDMARKS)
            .map(|p| {
                let (x, y) = crop.to_image(p[0], p[1]);
                [x, y, p[2] * scale]
            })
            .collect();
        Ok(Some(Mesh { points }))
    }
}

/// The eye line's angle, from the outer eye corners.
fn eye_roll(points: &[Point]) -> f32 {
    let (a, b) = (points[33], points[263]);
    let (left, right) = if a[0] <= b[0] { (a, b) } else { (b, a) };
    (right[1] - left[1]).atan2(right[0] - left[0])
}

/// Fills `input` (NHWC RGB, 0 to 1) with `crop` of `frame`, bilinear;
/// outside the frame reads black.
fn sample(frame: &Frame, crop: Crop, input: &mut [f32]) {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let pixel = |x: usize, y: usize, channel: usize| -> f32 {
        f32::from(frame.bgra[(y * w + x) * 4 + channel]) / 255.0
    };
    for v in 0..SIZE {
        for u in 0..SIZE {
            let (x, y) = crop.to_image(u as f32 + 0.5, v as f32 + 0.5);
            let (x, y) = (x - 0.5, y - 0.5);
            let out = &mut input[(v * SIZE + u) * 3..][..3];
            if x < 0.0 || y < 0.0 || x >= (w - 1) as f32 || y >= (h - 1) as f32 {
                out.fill(0.0);
                continue;
            }
            let (x0, y0) = (x as usize, y as usize);
            let (fx, fy) = (x - x0 as f32, y - y0 as f32);
            // BGRA in, RGB out.
            for (rgb, bgra) in [(0, 2), (1, 1), (2, 0)] {
                let top = pixel(x0, y0, bgra) * (1.0 - fx) + pixel(x0 + 1, y0, bgra) * fx;
                let bottom =
                    pixel(x0, y0 + 1, bgra) * (1.0 - fx) + pixel(x0 + 1, y0 + 1, bgra) * fx;
                out[rgb] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
}

/// The model's path, downloaded and checked against its pin first if needed.
fn ensure_model() -> Result<PathBuf> {
    let folder = crate::app_paths::support_dir()?.join("models").join("gaze");
    let path = folder.join("face_landmarks.onnx");
    if path.exists() {
        return Ok(path);
    }
    std::fs::create_dir_all(&folder)?;
    let partial = folder.join("face_landmarks.onnx.partial");
    eprintln!("Downloading Face Mesh (4.9 MB)…");
    crate::net::download(
        &crate::net::Download {
            url: MODEL_URL,
            user_agent: &format!("hex/{}", crate::VERSION),
            size: Some(MODEL_BYTES),
            canceled: None,
        },
        &partial,
        |_| {},
    )
    .wrap_err("Face Mesh didn't download")?;
    let digest = crate::transcription::models::sha256_hex(&partial, None)?;
    if digest != MODEL_SHA256 {
        let _ = std::fs::remove_file(&partial);
        bail!("Face Mesh's download didn't match its pin (got {digest})");
    }
    std::fs::rename(&partial, &path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_level_crop_maps_its_corners_onto_the_box() {
        let crop = Crop {
            cx: 100.0,
            cy: 50.0,
            side: 64.0,
            angle: 0.0,
        };
        assert_eq!(crop.to_image(0.0, 0.0), (68.0, 18.0));
        assert_eq!(crop.to_image(SIZE as f32, SIZE as f32), (132.0, 82.0));
    }

    #[test]
    fn a_turned_crop_turns_about_its_centre() {
        let crop = Crop {
            cx: 0.0,
            cy: 0.0,
            side: SIZE as f32,
            angle: std::f32::consts::FRAC_PI_2,
        };
        let (x, y) = crop.to_image(SIZE as f32, SIZE as f32 / 2.0);
        assert!(x.abs() < 1e-4 && (y - 128.0).abs() < 1e-4, "{x} {y}");
    }

    #[test]
    fn sampling_reads_rgb_from_bgra() {
        let mut frame = super::super::camera::Frame::new();
        frame.width = 4;
        frame.height = 4;
        for pixel in frame.bgra[..64].as_chunks_mut::<4>().0 {
            pixel.copy_from_slice(&[255, 0, 51, 255]);
        }
        let crop = Crop {
            cx: 2.0,
            cy: 2.0,
            side: 2.0,
            angle: 0.0,
        };
        let mut input = vec![0.0; SIZE * SIZE * 3];
        sample(&frame, crop, &mut input);
        let centre = &input[(128 * SIZE + 128) * 3..][..3];
        assert!((centre[0] - 0.2).abs() < 1e-6 && centre[1] == 0.0 && centre[2] == 1.0);
    }
}
