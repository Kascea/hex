//! One person's map from [`Features`] to a screen point: ridge regression
//! fitted while they look at known dots. It holds weights only, never
//! images, and is saved owner-only as `gaze-calibration.json`.

use std::path::PathBuf;

use color_eyre::eyre::Result;
use serde::{Deserialize, Serialize};

use super::features::Features;

/// How strongly weights are pulled towards zero, on standardized features.
pub(crate) const RIDGE: f64 = 0.01;

/// One moment of looking at a known point.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sample {
    pub features: Features,
    /// Where the person was looking: points in Rune's global space.
    pub target: (f64, f64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Calibration {
    /// Each term's mean and spread over the fitted samples (bias excluded).
    mean: Vec<f64>,
    spread: Vec<f64>,
    /// Weights for x and y, bias first.
    x: Vec<f64>,
    y: Vec<f64>,
}

/// The regression's terms for one sample, without the bias: the iris
/// offsets with their squares and product (gaze is not linear in them at the
/// screen's edges), then head pose and placement.
fn terms(f: &Features) -> [f64; 10] {
    let (ix, iy) = (f64::from(f.iris_x), f64::from(f.iris_y));
    [
        ix,
        iy,
        ix * ix,
        iy * iy,
        ix * iy,
        f64::from(f.yaw),
        f64::from(f.pitch),
        f64::from(f.face_x),
        f64::from(f.face_y),
        f64::from(f.scale),
    ]
}

const TERMS: usize = 10;

impl Calibration {
    /// Fits a calibration, or `None` without enough samples to fit one.
    pub(crate) fn fit(samples: &[Sample], ridge: f64) -> Option<Self> {
        if samples.len() <= TERMS + 1 {
            return None;
        }
        let rows: Vec<[f64; TERMS]> = samples.iter().map(|s| terms(&s.features)).collect();
        let n = rows.len() as f64;
        let mut mean = vec![0.0; TERMS];
        let mut spread = vec![0.0; TERMS];
        for row in &rows {
            for (m, value) in mean.iter_mut().zip(row) {
                *m += value / n;
            }
        }
        for row in &rows {
            for ((s, m), value) in spread.iter_mut().zip(&mean).zip(row) {
                *s += (value - m).powi(2) / n;
            }
        }
        // A term that never varied carries nothing; leave it at zero.
        for s in &mut spread {
            *s = if *s > 1e-12 { s.sqrt() } else { 0.0 };
        }
        let standard = |row: &[f64; TERMS]| -> Vec<f64> {
            std::iter::once(1.0)
                .chain((0..TERMS).map(|i| {
                    if spread[i] == 0.0 {
                        0.0
                    } else {
                        (row[i] - mean[i]) / spread[i]
                    }
                }))
                .collect()
        };
        let size = TERMS + 1;
        let mut normal = vec![vec![0.0; size]; size];
        let mut bx = vec![0.0; size];
        let mut by = vec![0.0; size];
        for (row, sample) in rows.iter().zip(samples) {
            let z = standard(row);
            for i in 0..size {
                for j in 0..size {
                    normal[i][j] += z[i] * z[j];
                }
                bx[i] += z[i] * sample.target.0;
                by[i] += z[i] * sample.target.1;
            }
        }
        // The bias isn't pulled towards zero; a dead term is pinned there.
        for (i, row) in normal.iter_mut().enumerate().skip(1) {
            row[i] += if spread[i - 1] == 0.0 { 1.0 } else { ridge * n };
        }
        Some(Self {
            x: solve(normal.clone(), bx)?,
            y: solve(normal, by)?,
            mean,
            spread,
        })
    }

    /// Where these features say the person is looking.
    pub(crate) fn predict(&self, features: &Features) -> (f64, f64) {
        let mut point = (self.x[0], self.y[0]);
        for (i, value) in terms(features).into_iter().enumerate() {
            if self.spread[i] == 0.0 {
                continue;
            }
            let z = (value - self.mean[i]) / self.spread[i];
            point.0 += self.x[i + 1] * z;
            point.1 += self.y[i + 1] * z;
        }
        point
    }

    pub(crate) fn path() -> Result<PathBuf> {
        Ok(crate::app_paths::support_dir()?.join("gaze-calibration.json"))
    }

    pub(crate) fn save(&self) -> Result<PathBuf> {
        let path = Self::path()?;
        crate::durable::write(&path, &serde_json::to_vec_pretty(self)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
        }
        Ok(path)
    }

    pub(crate) fn load() -> Result<Self> {
        Ok(serde_json::from_slice(&std::fs::read(Self::path()?)?)?)
    }
}

/// How far each sample's prediction lands from its target, in points.
pub(crate) fn errors(calibration: &Calibration, samples: &[Sample]) -> Vec<f64> {
    samples
        .iter()
        .map(|sample| {
            let (x, y) = calibration.predict(&sample.features);
            (x - sample.target.0).hypot(y - sample.target.1)
        })
        .collect()
}

/// The value `share` of the way up `values` (0.5 is the median).
pub(crate) fn percentile(values: &[f64], share: f64) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let last = sorted.len().checked_sub(1)?;
    Some(sorted[((last as f64) * share).round() as usize])
}

/// Solves `a x = b` by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for column in 0..n {
        let pivot =
            (column..n).max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))?;
        if a[pivot][column].abs() < 1e-12 {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        let (above, below) = a.split_at_mut(column + 1);
        let pivot_row = &above[column];
        for (offset, row) in below.iter_mut().enumerate() {
            let factor = row[column] / pivot_row[column];
            for (value, pivot) in row[column..].iter_mut().zip(&pivot_row[column..]) {
                *value -= factor * pivot;
            }
            b[column + 1 + offset] -= factor * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let rest: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - rest) / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Samples on a 5 by 5 grid of gazes whose screen point is a known
    /// function of the features, with the head moving a little.
    fn samples(screen: impl Fn(&Features) -> (f64, f64)) -> Vec<Sample> {
        let mut samples = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..4 {
                    let features = Features {
                        iris_x: -0.2 + 0.1 * i as f32,
                        iris_y: -0.1 + 0.05 * j as f32,
                        yaw: 0.01 * k as f32,
                        pitch: 0.6 + 0.005 * (i + k) as f32,
                        face_x: 0.5 + 0.01 * (j + k) as f32,
                        face_y: 0.4,
                        scale: 0.1,
                        ..Features::default()
                    };
                    samples.push(Sample {
                        target: screen(&features),
                        features,
                    });
                }
            }
        }
        samples
    }

    fn quadratic(f: &Features) -> (f64, f64) {
        let (ix, iy) = (f64::from(f.iris_x), f64::from(f.iris_y));
        (
            760.0 + 4000.0 * ix + 900.0 * ix * ix - 2000.0 * f64::from(f.yaw),
            480.0 + 5000.0 * iy + 300.0 * ix * iy,
        )
    }

    #[test]
    fn it_recovers_a_map_it_can_express() {
        let samples = samples(quadratic);
        let calibration = Calibration::fit(&samples, 1e-9).unwrap();
        let worst = percentile(&errors(&calibration, &samples), 1.0).unwrap();
        assert!(worst < 0.5, "worst error {worst} pt");
    }

    #[test]
    fn ridge_keeps_it_close_on_noisy_targets() {
        let mut samples = samples(quadratic);
        for (index, sample) in samples.iter_mut().enumerate() {
            let jitter = if index % 2 == 0 { 15.0 } else { -15.0 };
            sample.target.0 += jitter;
        }
        let calibration = Calibration::fit(&samples, RIDGE).unwrap();
        let median = percentile(&errors(&calibration, &samples), 0.5).unwrap();
        assert!(median < 25.0, "median error {median} pt");
    }

    #[test]
    fn too_few_samples_fit_nothing() {
        assert!(Calibration::fit(&samples(quadratic)[..TERMS], RIDGE).is_none());
    }

    #[test]
    fn a_saved_calibration_reads_back_the_same() {
        let calibration = Calibration::fit(&samples(quadratic), RIDGE).unwrap();
        let json = serde_json::to_vec(&calibration).unwrap();
        let read: Calibration = serde_json::from_slice(&json).unwrap();
        let features = samples(quadratic)[7].features;
        let (a, b) = (calibration.predict(&features), read.predict(&features));
        assert!((a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);
    }

    #[test]
    fn percentiles_pick_from_the_sorted_values() {
        let values = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(percentile(&values, 0.5), Some(3.0));
        assert_eq!(percentile(&values, 0.9), Some(5.0));
        assert_eq!(percentile(&[], 0.5), None);
    }
}
