//! `hex dev gaze`: phase 0's measurement. `probe` reads the camera and Face
//! Mesh and prints their rate and cost; `calibrate` shows dots on the
//! primary display, fits a calibration, measures it on dots it didn't see,
//! and reports against the plan's go/no-go targets; `track` draws the gaze.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use clap::Subcommand;
use color_eyre::eyre::{Result, bail, eyre};
use rune::{
    App, AppSpec, Cx, DisplayTarget, El, OverlayHandle, OverlayOptions, OverlayPlacement,
    OverlaySpec, RunOptions, WindowSpec, el, text,
};

use super::calibration::{self, Calibration, RIDGE, Sample};
use super::camera::{Camera, Display, Frame};
use super::features::{Features, features};
use super::fixation::{Fixations, SMOOTHING, Smoother};
use super::landmarks::FaceMesh;

#[derive(Subcommand)]
pub(crate) enum GazeCommand {
    /// Read the camera and Face Mesh, printing rate, cost, and features each second.
    Probe {
        #[arg(long, default_value_t = 20)]
        seconds: u64,
    },
    /// Fit a calibration on 9 dots, measure it on 5 others, and save it.
    Calibrate,
    /// Draw where the saved calibration says you're looking.
    Track {
        #[arg(long, default_value_t = 30)]
        seconds: u64,
    },
}

pub(crate) fn run(command: GazeCommand) -> Result<()> {
    match command {
        GazeCommand::Probe { seconds } => probe(Duration::from_secs(seconds)),
        GazeCommand::Calibrate => calibrate(),
        GazeCommand::Track { seconds } => track(Duration::from_secs(seconds)),
    }
}

/// Lid gap below this, in eye widths, is a blink: the iris reads wrong.
const BLINK: f32 = 0.12;
/// How long a dot shows before its samples count, while the eyes land.
const SETTLE: Duration = Duration::from_millis(900);
const COLLECT: Duration = Duration::from_millis(1200);
/// Calibration dots, as shares of the display: a 3 by 3 grid, snaking.
const FIT_DOTS: [(f64, f64); 9] = [
    (0.1, 0.1),
    (0.5, 0.1),
    (0.9, 0.1),
    (0.9, 0.5),
    (0.5, 0.5),
    (0.1, 0.5),
    (0.1, 0.9),
    (0.5, 0.9),
    (0.9, 0.9),
];
/// Validation dots the fit never saw.
const CHECK_DOTS: [(f64, f64); 5] = [(0.3, 0.3), (0.7, 0.3), (0.5, 0.25), (0.3, 0.7), (0.7, 0.7)];

// The plan's go/no-go targets.
const TARGET_MEDIAN_CM: f64 = 3.0;
const TARGET_P90_CM: f64 = 5.0;
const TARGET_RATE: f64 = 15.0;
const TARGET_CORE: f64 = 0.15;

/// One face's features, as the tracker read them.
#[derive(Clone, Copy)]
struct Observation {
    at: Instant,
    features: Features,
}

/// What the tracker has measured so far.
#[derive(Default)]
struct Stats {
    frames: u64,
    faces: u64,
    meshes: u64,
    inference_ms: Vec<f64>,
    thread_cpu_s: f64,
}

/// The camera and Face Mesh on one thread, sending each face's features.
/// Stops and joins when dropped.
struct Tracker {
    observations: Receiver<Observation>,
    stats: Arc<Mutex<Stats>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    started: Instant,
    camera_name: String,
}

impl Tracker {
    fn start() -> Result<Self> {
        let mesh = FaceMesh::load()?;
        let camera = Camera::start()?;
        let camera_name = camera.name();
        let (sender, observations) = sync_channel(64);
        let stats = Arc::new(Mutex::new(Stats::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = std::thread::Builder::new().name("hex-gaze".into()).spawn({
            let (stats, stop) = (stats.clone(), stop.clone());
            move || track_frames(camera, mesh, &sender, &stats, &stop)
        })?;
        Ok(Self {
            observations,
            stats,
            stop,
            thread: Some(thread),
            started: Instant::now(),
            camera_name,
        })
    }

    /// Observations from now until `until`, blinks left out.
    fn collect(&self, until: Instant) -> Vec<Observation> {
        let mut seen = Vec::new();
        loop {
            let left = until.saturating_duration_since(Instant::now());
            match self.observations.recv_timeout(left) {
                Ok(observation) if observation.features.openness >= BLINK => seen.push(observation),
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => return seen,
            }
        }
    }

    fn drain(&self) {
        while self.observations.try_recv().is_ok() {}
    }

    /// Camera rate, face share, mesh rate, inference median and 90th
    /// percentile, and the tracker thread's share of one core.
    fn summary(&self) -> Summary {
        let stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        let seconds = self.started.elapsed().as_secs_f64().max(1e-6);
        Summary {
            camera_rate: stats.frames as f64 / seconds,
            face_share: stats.faces as f64 / (stats.frames.max(1)) as f64,
            mesh_rate: stats.meshes as f64 / seconds,
            inference_median: calibration::percentile(&stats.inference_ms, 0.5).unwrap_or(0.0),
            inference_p90: calibration::percentile(&stats.inference_ms, 0.9).unwrap_or(0.0),
            core: stats.thread_cpu_s / seconds,
        }
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct Summary {
    camera_rate: f64,
    face_share: f64,
    mesh_rate: f64,
    inference_median: f64,
    inference_p90: f64,
    core: f64,
}

fn track_frames(
    camera: Camera,
    mut mesh: FaceMesh,
    sender: &SyncSender<Observation>,
    stats: &Mutex<Stats>,
    stop: &AtomicBool,
) {
    let cpu_start = thread_cpu_seconds();
    let mut frame = Frame::new();
    while !stop.load(Ordering::Relaxed) {
        match camera.next(&mut frame, Duration::from_millis(200)) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(error) => {
                eprintln!("{error}");
                return;
            }
        }
        let at = Instant::now();
        let mut found = None;
        let mut inference = None;
        if let Some(face) = frame.face {
            let started = Instant::now();
            match mesh.detect(&frame, face) {
                Ok(points) => found = points,
                Err(error) => eprintln!("Face Mesh failed: {error}"),
            }
            inference = Some(started.elapsed().as_secs_f64() * 1000.0);
        }
        let features = found
            .as_ref()
            .and_then(|mesh| features(&mesh.points, (frame.width, frame.height)));
        {
            let mut stats = stats.lock().unwrap_or_else(|e| e.into_inner());
            stats.frames += 1;
            stats.faces += u64::from(frame.face.is_some());
            stats.meshes += u64::from(features.is_some());
            stats.inference_ms.extend(inference);
            stats.thread_cpu_s = thread_cpu_seconds() - cpu_start;
        }
        if let Some(features) = features {
            // A full queue means nobody is reading yet; drop the frame.
            if let Err(TrySendError::Disconnected(_)) =
                sender.try_send(Observation { at, features })
            {
                return;
            }
        }
    }
}

fn thread_cpu_seconds() -> f64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    time.tv_sec as f64 + time.tv_nsec as f64 / 1e9
}

fn probe(duration: Duration) -> Result<()> {
    let tracker = Tracker::start()?;
    println!("Camera: {}. Ctrl-C stops.", tracker.camera_name);
    let end = Instant::now() + duration;
    while Instant::now() < end {
        let second = tracker.collect(Instant::now() + Duration::from_secs(1));
        let s = tracker.summary();
        let last = second.last().map(|o| o.features);
        print!(
            "{:5.1} frames/s, face {:3.0}%, mesh {:5.1}/s, Face Mesh {:4.1} ms (p90 {:4.1}), gaze thread {:3.0}% of a core",
            s.camera_rate,
            s.face_share * 100.0,
            s.mesh_rate,
            s.inference_median,
            s.inference_p90,
            s.core * 100.0
        );
        match last {
            Some(f) => println!(
                " | iris {:+.3} {:+.3}, yaw {:+.2}, pitch {:+.2}, roll {:+.1}°, open {:.2}",
                f.iris_x,
                f.iris_y,
                f.yaw,
                f.pitch,
                f.roll.to_degrees(),
                f.openness
            ),
            None => println!(" | no face"),
        }
    }
    Ok(())
}

/// What the overlay draws, in points from the primary display's top left.
#[derive(Default)]
struct Dots {
    target: Option<(f32, f32)>,
    /// The target dot shrinks while its samples count.
    collecting: bool,
    gaze: Option<(f32, f32)>,
    note: String,
}

const TARGET_DOT: f32 = 28.0;
const COLLECTING_DOT: f32 = 12.0;
const GAZE_DOT: f32 = 36.0;

fn view(dots: &Dots, cx: &mut Cx<Dots>) -> El {
    let k = rune_theme_opencode::kit(cx.system_dark());
    let dot = |(x, y): (f32, f32), size: f32, key: &str| {
        el().key(key)
            .absolute(x - size / 2.0, y - size / 2.0)
            .size(size, size)
            .radius(size / 2.0)
    };
    let mut root = el().key("gaze").w_full().h_full().fill(k.c.scrim).child(
        text(dots.note.clone())
            .key("note")
            .absolute(32.0, 32.0)
            .font(k.ty.ui)
            .color(k.c.text_base),
    );
    if let Some(target) = dots.target {
        let size = if dots.collecting {
            COLLECTING_DOT
        } else {
            TARGET_DOT
        };
        // White in either appearance: the scrim is dark in both, and a
        // theme's text colour is dark in light mode.
        root = root.child(dot(target, size, "target").fill(rune::Color::WHITE));
    }
    if let Some(gaze) = dots.gaze {
        root = root.child(dot(gaze, GAZE_DOT, "gaze").fill(k.c.bg_accent));
    }
    root
}

/// Shows a full-screen overlay on the primary display and runs `script` on
/// its own thread, quitting when it returns.
fn with_overlay(
    script: impl FnOnce(OverlayHandle<Dots>) -> Result<()> + Send + 'static,
) -> Result<()> {
    let outcome: Arc<Mutex<Option<Result<()>>>> = Arc::default();
    let result = outcome.clone();
    let spec = AppSpec {
        fonts: rune_theme_opencode::FONTS,
        families: rune_theme_opencode::FAMILIES,
        ..AppSpec::new("Hex gaze", (320.0, 200.0), Dots::default, view)
    };
    rune::run_windows_with_args(RunOptions { accessory: true }, [], move |_| {
        let mut app = App::new(spec).windowless();
        let overlay = app.overlay(
            OverlaySpec {
                name: "Hex gaze",
                size: (320.0, 200.0),
                edge: rune::ScreenEdge::Top,
                margin: 0.0,
                align: rune::Align::Center,
                inset: 0.0,
                interactive: false,
                view,
            }
            .options(OverlayOptions {
                placement: Some(OverlayPlacement::FullScreen),
                display: DisplayTarget::Primary,
                ..Default::default()
            }),
            Dots::default(),
        );
        overlay.show();
        let handle = app.handle();
        std::thread::spawn(move || {
            let done = script(overlay);
            *result.lock().unwrap_or_else(|e| e.into_inner()) = Some(done);
            handle.quit();
        });
        vec![WindowSpec::from_app("gaze", app)]
    });
    outcome
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_else(|| Err(eyre!("the overlay closed before it finished")))
}

fn show(overlay: &OverlayHandle<Dots>, change: impl FnOnce(&mut Dots) + Send + 'static) {
    overlay.update(move |dots, _| change(dots));
}

/// Shows each dot, settles, and collects its samples.
fn run_dots(
    tracker: &Tracker,
    overlay: &OverlayHandle<Dots>,
    display: &Display,
    dots: &[(f64, f64)],
    note: &str,
) -> Vec<Vec<Sample>> {
    dots.iter()
        .enumerate()
        .map(|(index, (fx, fy))| {
            let local = (display.width * fx, display.height * fy);
            let note = format!(
                "{note} ({}/{}). Keep your head still.",
                index + 1,
                dots.len()
            );
            show(overlay, move |d| {
                d.target = Some((local.0 as f32, local.1 as f32));
                d.collecting = false;
                d.note = note;
            });
            std::thread::sleep(SETTLE);
            tracker.drain();
            show(overlay, |d| d.collecting = true);
            let target = (display.x + local.0, display.y + local.1);
            tracker
                .collect(Instant::now() + COLLECT)
                .into_iter()
                .map(|o| Sample {
                    features: o.features,
                    target,
                })
                .collect()
        })
        .collect()
}

fn calibrate() -> Result<()> {
    let display = Display::primary();
    let tracker = Tracker::start()?;
    println!(
        "Camera: {}. Look at each dot until the next one shows.",
        tracker.camera_name
    );
    let report: Arc<Mutex<Option<Report>>> = Arc::default();
    let out = report.clone();
    with_overlay(move |overlay| {
        show(&overlay, |d| {
            d.note = "Sit as you normally do. Look at each dot.".into();
        });
        std::thread::sleep(Duration::from_secs(2));
        let fit = run_dots(&tracker, &overlay, &display, &FIT_DOTS, "Calibrating");
        let check = run_dots(&tracker, &overlay, &display, &CHECK_DOTS, "Checking");
        *out.lock().unwrap_or_else(|e| e.into_inner()) =
            Some(Report::new(&fit, &check, tracker.summary()));
        Ok(())
    })?;
    let report = report
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .ok_or_else(|| eyre!("calibration didn't finish"))?;
    report.print(&display)?;
    Ok(())
}

struct Report {
    calibration: Option<Calibration>,
    fit_samples: usize,
    check_samples: usize,
    /// Each validation sample's error, raw and after smoothing, in points.
    raw: Vec<f64>,
    smoothed: Vec<f64>,
    summary: Summary,
}

impl Report {
    fn new(fit: &[Vec<Sample>], check: &[Vec<Sample>], summary: Summary) -> Self {
        let fit_samples: Vec<Sample> = fit.iter().flatten().copied().collect();
        let calibration = Calibration::fit(&fit_samples, RIDGE);
        let mut raw = Vec::new();
        let mut smoothed = Vec::new();
        if let Some(calibration) = &calibration {
            for dot in check {
                raw.extend(calibration::errors(calibration, dot));
                // Each dot's samples in order through a fresh smoother, as a
                // fixation would see them; its first few are still settling.
                let mut smoother = Smoother::new(SMOOTHING);
                for (index, sample) in dot.iter().enumerate() {
                    let point = smoother.push(
                        index as f64 / summary.mesh_rate.max(1.0),
                        calibration.predict(&sample.features),
                    );
                    if index >= 5 {
                        smoothed.push((point.0 - sample.target.0).hypot(point.1 - sample.target.1));
                    }
                }
            }
        }
        Self {
            calibration,
            fit_samples: fit_samples.len(),
            check_samples: check.iter().map(Vec::len).sum(),
            raw,
            smoothed,
            summary,
        }
    }

    fn print(&self, display: &Display) -> Result<()> {
        let s = &self.summary;
        println!(
            "Camera {:.1} frames/s, face in {:.0}%, gaze {:.1}/s. Face Mesh {:.1} ms median, {:.1} ms p90. Gaze thread {:.0}% of one core.",
            s.camera_rate,
            s.face_share * 100.0,
            s.mesh_rate,
            s.inference_median,
            s.inference_p90,
            s.core * 100.0
        );
        println!(
            "{} samples fitted, {} checked.",
            self.fit_samples, self.check_samples
        );
        let Some(calibration) = &self.calibration else {
            bail!("Too few samples to fit a calibration: is the camera seeing your face?");
        };
        let cm = display.cm_per_point();
        let line = |name: &str, errors: &[f64]| -> Option<(f64, f64)> {
            let median = calibration::percentile(errors, 0.5)?;
            let p90 = calibration::percentile(errors, 0.9)?;
            match cm {
                Some(cm) => println!(
                    "{name}: median {median:.0} pt ({:.1} cm), p90 {p90:.0} pt ({:.1} cm)",
                    median * cm,
                    p90 * cm
                ),
                None => println!("{name}: median {median:.0} pt, p90 {p90:.0} pt"),
            }
            Some((median, p90))
        };
        line("Error, raw", &self.raw);
        let smoothed = line("Error, smoothed", &self.smoothed);
        if let (Some(cm), Some((median, p90))) = (cm, smoothed) {
            let verdict = |pass: bool| if pass { "pass" } else { "FAIL" };
            println!(
                "Targets: median ≤ {TARGET_MEDIAN_CM} cm {}, p90 ≤ {TARGET_P90_CM} cm {}, ≥ {TARGET_RATE}/s {}, ≤ {:.0}% of a core {}",
                verdict(median * cm <= TARGET_MEDIAN_CM),
                verdict(p90 * cm <= TARGET_P90_CM),
                verdict(s.mesh_rate >= TARGET_RATE),
                TARGET_CORE * 100.0,
                verdict(s.core <= TARGET_CORE),
            );
        }
        println!("Saved {}", calibration.save()?.display());
        Ok(())
    }
}

fn track(duration: Duration) -> Result<()> {
    let calibration = Calibration::load()
        .map_err(|error| eyre!("No calibration ({error}); run `hex dev gaze calibrate` first."))?;
    let display = Display::primary();
    let tracker = Tracker::start()?;
    let radius = display.cm_per_point().map_or(60.0, |cm| 1.5 / cm);
    with_overlay(move |overlay| {
        let mut smoother = Smoother::new(SMOOTHING);
        let mut fixations = Fixations::new(radius, 0.15);
        let started = Instant::now();
        show(&overlay, |d| {
            d.note = "Look around; the dot follows your gaze.".into();
        });
        while started.elapsed() < duration {
            for observation in tracker.collect(Instant::now() + Duration::from_millis(30)) {
                let time = observation.at.duration_since(started).as_secs_f64();
                let point = smoother.push(time, calibration.predict(&observation.features));
                if let Some(fixation) = fixations.push(time, point) {
                    println!(
                        "{time:6.2}s fixation at {:.0}, {:.0}",
                        fixation.centre.0, fixation.centre.1
                    );
                }
                let local = ((point.0 - display.x) as f32, (point.1 - display.y) as f32);
                show(&overlay, move |d| d.gaze = Some(local));
            }
        }
        Ok(())
    })
}
