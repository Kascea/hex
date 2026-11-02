//! Smoothing gaze and finding where it rests. The One Euro filter steadies a
//! still gaze without lagging a moving one; a fixation is the gaze staying
//! inside a circle for long enough.

use std::f64::consts::TAU;

/// The One Euro filter's settings: slower changes than `min_cutoff` (Hz) are
/// kept, and the cutoff rises by `beta` per point a second of movement.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Smoothing {
    pub min_cutoff: f64,
    pub beta: f64,
}

pub(crate) const SMOOTHING: Smoothing = Smoothing {
    min_cutoff: 1.0,
    beta: 0.01,
};
/// The cutoff for the speed estimate itself.
const SPEED_CUTOFF: f64 = 1.0;

/// One Euro filters for x and y.
#[derive(Clone, Debug)]
pub(crate) struct Smoother {
    settings: Smoothing,
    last: Option<Last>,
}

#[derive(Clone, Copy, Debug)]
struct Last {
    time: f64,
    point: (f64, f64),
    speed: (f64, f64),
}

impl Smoother {
    pub(crate) fn new(settings: Smoothing) -> Self {
        Self {
            settings,
            last: None,
        }
    }

    /// The smoothed point for `point` seen at `time` seconds.
    pub(crate) fn push(&mut self, time: f64, point: (f64, f64)) -> (f64, f64) {
        let Some(last) = self.last.filter(|last| time > last.time) else {
            self.last = Some(Last {
                time,
                point,
                speed: (0.0, 0.0),
            });
            return point;
        };
        let dt = time - last.time;
        let axis = |value: f64, previous: f64, previous_speed: f64| {
            let speed = blend(
                (value - previous) / dt,
                previous_speed,
                alpha(SPEED_CUTOFF, dt),
            );
            let cutoff = self.settings.min_cutoff + self.settings.beta * speed.abs();
            (blend(value, previous, alpha(cutoff, dt)), speed)
        };
        let (x, sx) = axis(point.0, last.point.0, last.speed.0);
        let (y, sy) = axis(point.1, last.point.1, last.speed.1);
        self.last = Some(Last {
            time,
            point: (x, y),
            speed: (sx, sy),
        });
        (x, y)
    }
}

fn alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (TAU * cutoff);
    1.0 / (1.0 + tau / dt)
}

fn blend(value: f64, previous: f64, alpha: f64) -> f64 {
    alpha * value + (1.0 - alpha) * previous
}

/// Where the gaze rested.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Fixation {
    pub centre: (f64, f64),
    pub start: f64,
}

/// Finds fixations: the gaze staying within `radius` points of where it
/// settled for at least `duration` seconds. Each fixation is reported once.
#[derive(Clone, Debug)]
pub(crate) struct Fixations {
    radius: f64,
    duration: f64,
    run: Vec<(f64, (f64, f64))>,
    reported: bool,
}

impl Fixations {
    pub(crate) fn new(radius: f64, duration: f64) -> Self {
        Self {
            radius,
            duration,
            run: Vec::new(),
            reported: false,
        }
    }

    /// Adds a (smoothed) point; returns a fixation the first time the
    /// current run lasts long enough.
    pub(crate) fn push(&mut self, time: f64, point: (f64, f64)) -> Option<Fixation> {
        let centre = self.centre();
        if centre.is_some_and(|c| (point.0 - c.0).hypot(point.1 - c.1) > self.radius) {
            self.run.clear();
            self.reported = false;
        }
        self.run.push((time, point));
        let start = self.run[0].0;
        if self.reported || time - start < self.duration {
            return None;
        }
        self.reported = true;
        Some(Fixation {
            centre: self.centre()?,
            start,
        })
    }

    fn centre(&self) -> Option<(f64, f64)> {
        let n = self.run.len() as f64;
        (n > 0.0).then(|| {
            let (x, y) = self
                .run
                .iter()
                .fold((0.0, 0.0), |(x, y), (_, p)| (x + p.0, y + p.1));
            (x / n, y / n)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_still_gaze_settles_where_it_is() {
        let mut smoother = Smoother::new(SMOOTHING);
        let mut last = (0.0, 0.0);
        for i in 0..60 {
            let wobble = if i % 2 == 0 { 20.0 } else { -20.0 };
            last = smoother.push(f64::from(i) / 30.0, (500.0 + wobble, 300.0));
        }
        assert!((last.0 - 500.0).abs() < 8.0, "{last:?}");
        assert!((last.1 - 300.0).abs() < 1e-9);
    }

    #[test]
    fn smoothing_follows_a_jump() {
        let mut smoother = Smoother::new(SMOOTHING);
        for i in 0..30 {
            smoother.push(f64::from(i) / 30.0, (100.0, 100.0));
        }
        let mut last = (0.0, 0.0);
        for i in 30..45 {
            last = smoother.push(f64::from(i) / 30.0, (900.0, 100.0));
        }
        assert!(last.0 > 800.0, "{last:?}");
    }

    #[test]
    fn a_fixation_is_reported_once_after_its_duration() {
        let mut fixations = Fixations::new(40.0, 0.15);
        let mut found = Vec::new();
        for i in 0..20 {
            let time = f64::from(i) / 30.0;
            let jitter = if i % 2 == 0 { 10.0 } else { -10.0 };
            if let Some(fixation) = fixations.push(time, (400.0 + jitter, 200.0)) {
                found.push((time, fixation));
            }
        }
        assert_eq!(found.len(), 1);
        let (time, fixation) = found[0];
        assert!((0.15..0.2).contains(&time), "{time}");
        assert_eq!(fixation.start, 0.0);
        assert!((fixation.centre.0 - 400.0).abs() < 10.0);
    }

    #[test]
    fn moving_away_starts_a_new_fixation() {
        let mut fixations = Fixations::new(40.0, 0.15);
        let mut found = Vec::new();
        for i in 0..40 {
            let x = if i < 20 { 100.0 } else { 700.0 };
            found.extend(fixations.push(f64::from(i) / 30.0, (x, 200.0)));
        }
        assert_eq!(found.len(), 2);
        assert_eq!(found[1].centre, (700.0, 200.0));
        assert!((found[1].start - 20.0 / 30.0).abs() < 1e-9);
    }
}
