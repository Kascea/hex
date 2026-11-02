//! Face Mesh points to the numbers calibration maps to a screen point: where
//! each iris sits in its eye, and how the head is turned and placed. Pure
//! geometry in image pixels, y down, so it's the same for any camera.

use serde::{Deserialize, Serialize};

/// Points Face Mesh V2 returns: the 468-point mesh, then 10 iris points.
pub(crate) const LANDMARKS: usize = 478;

/// One landmark: x and y in image pixels, z relative depth in pixels.
pub(crate) type Point = [f32; 3];

/// Mesh indices of one eye: its two corners and its upper and lower lid.
struct Eye {
    corners: [usize; 2],
    lids: [usize; 2],
}

// MediaPipe's canonical mesh indices.
const EYES: [Eye; 2] = [
    Eye {
        corners: [33, 133],
        lids: [159, 145],
    },
    Eye {
        corners: [362, 263],
        lids: [386, 374],
    },
];
/// Each iris's centre point. Which eye each belongs to is decided by
/// distance, so a mirrored camera can't swap them.
const IRIS_CENTRES: [usize; 2] = [468, 473];
const NOSE_TIP: usize = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Features {
    /// The irises' mean offset from their eye's centre along the eye line
    /// (towards the image's right), in eye widths.
    pub iris_x: f32,
    /// The same across the eye line (towards the image's bottom).
    pub iris_y: f32,
    /// The nose tip's offset from the eyes' midpoint along the eye line, in
    /// distances between the eyes: turning the head left or right.
    pub yaw: f32,
    /// The same across the eye line: tilting the head up or down.
    pub pitch: f32,
    /// The eye line's angle from horizontal, radians.
    pub roll: f32,
    /// The eyes' midpoint in the frame, 0 to 1.
    pub face_x: f32,
    pub face_y: f32,
    /// The distance between the eyes as a share of the frame's width: how
    /// near the face is.
    pub scale: f32,
    /// The mean gap between the lids in eye widths; a blink reads near zero.
    pub openness: f32,
}

/// The features of one face, or `None` when its points are degenerate.
pub(crate) fn features(points: &[Point], frame: (u32, u32)) -> Option<Features> {
    if points.len() < LANDMARKS || frame.0 == 0 || frame.1 == 0 {
        return None;
    }
    let at = |index: usize| [points[index][0], points[index][1]];
    let centres = EYES.map(|eye| mid(at(eye.corners[0]), at(eye.corners[1])));
    // The eye line runs from the eye on the image's left to the one on its right.
    let (left, right) = if centres[0][0] <= centres[1][0] {
        (0, 1)
    } else {
        (1, 0)
    };
    let between = distance(centres[left], centres[right]);
    if between < 1.0 {
        return None;
    }
    let u = scale(sub(centres[right], centres[left]), 1.0 / between);
    let v = [-u[1], u[0]];
    let mut iris = [0.0f32; 2];
    let mut openness = 0.0;
    for (index, eye) in EYES.iter().enumerate() {
        let width = distance(at(eye.corners[0]), at(eye.corners[1]));
        if width < 1.0 {
            return None;
        }
        let centre = centres[index];
        let pupil = IRIS_CENTRES
            .map(at)
            .into_iter()
            .min_by(|a, b| distance(*a, centre).total_cmp(&distance(*b, centre)))?;
        let offset = sub(pupil, centre);
        iris[0] += dot(offset, u) / width / 2.0;
        iris[1] += dot(offset, v) / width / 2.0;
        openness += distance(at(eye.lids[0]), at(eye.lids[1])) / width / 2.0;
    }
    let middle = mid(centres[left], centres[right]);
    let nose = sub(at(NOSE_TIP), middle);
    Some(Features {
        iris_x: iris[0],
        iris_y: iris[1],
        yaw: dot(nose, u) / between,
        pitch: dot(nose, v) / between,
        roll: u[1].atan2(u[0]),
        face_x: middle[0] / frame.0 as f32,
        face_y: middle[1] / frame.1 as f32,
        scale: between / frame.0 as f32,
        openness,
    })
}

fn mid(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn scale(a: [f32; 2], by: f32) -> [f32; 2] {
    [a[0] * by, a[1] * by]
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: (u32, u32) = (640, 480);

    /// A level face: eyes 60 px apart around (320, 200), each 30 px wide,
    /// irises `iris` eye widths off centre, nose 40 px below the eyes.
    fn face(iris: [f32; 2]) -> Vec<Point> {
        let mut points = vec![[0.0; 3]; LANDMARKS];
        for (eye, cx) in EYES.iter().zip([290.0, 350.0]) {
            points[eye.corners[0]] = [cx - 15.0, 200.0, 0.0];
            points[eye.corners[1]] = [cx + 15.0, 200.0, 0.0];
            points[eye.lids[0]] = [cx, 194.0, 0.0];
            points[eye.lids[1]] = [cx, 206.0, 0.0];
        }
        points[IRIS_CENTRES[0]] = [290.0 + iris[0] * 30.0, 200.0 + iris[1] * 30.0, 0.0];
        points[IRIS_CENTRES[1]] = [350.0 + iris[0] * 30.0, 200.0 + iris[1] * 30.0, 0.0];
        points[NOSE_TIP] = [320.0, 240.0, 0.0];
        points
    }

    fn rotated(points: &[Point], angle: f32) -> Vec<Point> {
        let (sin, cos) = angle.sin_cos();
        points
            .iter()
            .map(|[x, y, z]| {
                let (dx, dy) = (x - 320.0, y - 200.0);
                [320.0 + dx * cos - dy * sin, 200.0 + dx * sin + dy * cos, *z]
            })
            .collect()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn a_centred_iris_on_a_level_face_reads_zero() {
        let f = features(&face([0.0, 0.0]), FRAME).unwrap();
        assert!(close(f.iris_x, 0.0) && close(f.iris_y, 0.0));
        assert!(close(f.roll, 0.0) && close(f.yaw, 0.0));
        assert!(close(f.pitch, 40.0 / 60.0));
        assert!(close(f.face_x, 0.5) && close(f.scale, 60.0 / 640.0));
        assert!(close(f.openness, 12.0 / 30.0));
    }

    #[test]
    fn iris_offsets_read_in_eye_widths() {
        let f = features(&face([0.1, -0.05]), FRAME).unwrap();
        assert!(close(f.iris_x, 0.1) && close(f.iris_y, -0.05));
    }

    #[test]
    fn tilting_the_head_changes_only_roll_and_placement() {
        let level = features(&face([0.1, 0.05]), FRAME).unwrap();
        let tilted = features(&rotated(&face([0.1, 0.05]), 0.3), FRAME).unwrap();
        assert!(close(tilted.roll, 0.3));
        assert!(close(tilted.iris_x, level.iris_x) && close(tilted.iris_y, level.iris_y));
        assert!(close(tilted.yaw, level.yaw) && close(tilted.pitch, level.pitch));
    }

    #[test]
    fn a_degenerate_face_has_no_features() {
        assert!(features(&vec![[0.0; 3]; LANDMARKS], FRAME).is_none());
        assert!(features(&face([0.0, 0.0])[..100], FRAME).is_none());
    }
}
