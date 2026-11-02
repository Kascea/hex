import AVFoundation
import CoreGraphics
import Foundation
import Vision

/// The default camera at 640×480 BGRA. Each frame gets Vision's largest face
/// box, then replaces the one before it: Rust reads only the latest frame,
/// and a frame it never read is dropped, never queued.
private final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "hex.gaze.camera", qos: .userInitiated)
    let faces = VNDetectFaceRectanglesRequest()
    let condition = NSCondition()
    var name = ""
    // Guarded by `condition`.
    var pixels = [UInt8]()
    var width = 0
    var height = 0
    var sequence: UInt64 = 0
    var time = 0.0
    var face: CGRect?

    func start() -> String? {
        guard let device = AVCaptureDevice.default(for: .video) else {
            return "This Mac has no camera."
        }
        name = device.localizedName
        let input: AVCaptureDeviceInput
        do {
            input = try AVCaptureDeviceInput(device: device)
        } catch {
            return "The camera didn't open: \(error.localizedDescription)"
        }
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        session.beginConfiguration()
        if session.canSetSessionPreset(.vga640x480) {
            session.sessionPreset = .vga640x480
        }
        guard session.canAddInput(input), session.canAddOutput(output) else {
            session.commitConfiguration()
            return "The camera is in use or can't give frames."
        }
        session.addInput(input)
        session.addOutput(output)
        session.commitConfiguration()
        session.startRunning()
        return nil
    }

    func stop() {
        session.stopRunning()
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up, options: [:])
        try? handler.perform([faces])
        let largest = (faces.results ?? []).max { $0.boundingBox.width < $1.boundingBox.width }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        var packed = [UInt8](repeating: 0, count: w * h * 4)
        packed.withUnsafeMutableBytes { out in
            for row in 0..<h {
                memcpy(out.baseAddress! + row * w * 4, base + row * stride, w * 4)
            }
        }
        // Vision's box is normalized with its origin at the bottom left.
        let box = largest.map { observation -> CGRect in
            let b = observation.boundingBox
            return CGRect(
                x: b.minX * CGFloat(w),
                y: (1 - b.maxY) * CGFloat(h),
                width: b.width * CGFloat(w),
                height: b.height * CGFloat(h)
            )
        }
        condition.lock()
        pixels = packed
        width = w
        height = h
        time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        face = box
        sequence &+= 1
        condition.broadcast()
        condition.unlock()
    }
}

private var camera: Camera?

private func cString(_ string: String) -> UnsafeMutablePointer<CChar>? {
    strdup(string)
}

/// 0 not determined, 1 restricted, 2 denied, 3 authorized.
@_cdecl("hex_gaze_camera_authorization")
public func hexGazeCameraAuthorization() -> Int32 {
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .notDetermined: return 0
    case .restricted: return 1
    case .denied: return 2
    case .authorized: return 3
    @unknown default: return 2
    }
}

/// Asks for Camera access and waits for the answer.
@_cdecl("hex_gaze_camera_request_access")
public func hexGazeCameraRequestAccess() -> Bool {
    let done = DispatchSemaphore(value: 0)
    var granted = false
    AVCaptureDevice.requestAccess(for: .video) { answer in
        granted = answer
        done.signal()
    }
    done.wait()
    return granted
}

/// Starts the camera. Returns 0, or 1 with `error` set (free it with
/// `hex_gaze_free_string`).
@_cdecl("hex_gaze_camera_start")
public func hexGazeCameraStart(_ error: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32 {
    if camera != nil { return 0 }
    let started = Camera()
    if let message = started.start() {
        error.pointee = cString(message)
        return 1
    }
    camera = started
    return 0
}

@_cdecl("hex_gaze_camera_stop")
public func hexGazeCameraStop() {
    camera?.stop()
    camera = nil
}

@_cdecl("hex_gaze_camera_name")
public func hexGazeCameraName() -> UnsafeMutablePointer<CChar>? {
    cString(camera?.name ?? "")
}

/// Waits up to `timeoutMs` for a frame newer than `after` and copies it into
/// `out`. `info` receives [sequence, width, height, time s, has face, face x,
/// y, width, height] (face in pixels, origin top left). Returns 1 when a
/// frame was copied, 0 on timeout, -1 when `out` is too small or the camera
/// isn't running.
@_cdecl("hex_gaze_camera_next")
public func hexGazeCameraNext(
    _ after: UInt64,
    _ timeoutMs: Int32,
    _ out: UnsafeMutablePointer<UInt8>,
    _ capacity: Int,
    _ info: UnsafeMutablePointer<Double>
) -> Int32 {
    guard let camera else { return -1 }
    let deadline = Date(timeIntervalSinceNow: Double(timeoutMs) / 1000)
    camera.condition.lock()
    defer { camera.condition.unlock() }
    while camera.sequence <= after {
        if !camera.condition.wait(until: deadline) { return 0 }
    }
    guard camera.pixels.count <= capacity else { return -1 }
    camera.pixels.withUnsafeBytes { bytes in
        out.update(from: bytes.bindMemory(to: UInt8.self).baseAddress!, count: bytes.count)
    }
    let face = camera.face
    let values: [Double] = [
        Double(camera.sequence), Double(camera.width), Double(camera.height), camera.time,
        face == nil ? 0 : 1,
        Double(face?.minX ?? 0), Double(face?.minY ?? 0),
        Double(face?.width ?? 0), Double(face?.height ?? 0),
    ]
    for (index, value) in values.enumerated() { info[index] = value }
    return 1
}

/// The primary display: [x, y, width, height] in points of the global
/// space (top left origin), then its physical [width, height] in mm (0 when
/// unknown).
@_cdecl("hex_gaze_primary_display")
public func hexGazePrimaryDisplay(_ out: UnsafeMutablePointer<Double>) {
    let display = CGMainDisplayID()
    let bounds = CGDisplayBounds(display)
    let size = CGDisplayScreenSize(display)
    let values = [bounds.minX, bounds.minY, bounds.width, bounds.height, size.width, size.height]
    for (index, value) in values.enumerated() { out[index] = Double(value) }
}

@_cdecl("hex_gaze_free_string")
public func hexGazeFreeString(_ string: UnsafeMutablePointer<CChar>?) {
    free(string)
}
