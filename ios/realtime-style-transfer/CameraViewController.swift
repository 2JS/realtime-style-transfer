import AVFoundation
import Combine
import CoreML
import Then
import UIKit

class CameraViewController: UIViewController {
    private let session = AVCaptureSession()
    private let videoStream = VideoStream()
    private let previewView = PreviewView()
    private let transferredView = UIImageView()

    let cicontext = CIContext()

    private var cancellableBag = Set<AnyCancellable>()

    override func viewDidLoad() {
        super.viewDidLoad()

        view.backgroundColor = .black

        view.addSubview(previewView)

        view.addSubview(transferredView)

        setupSession()

        videoStream.publisher
            .throttle(for: 0.066, scheduler: RunLoop.current, latest: true)
            .sink { [unowned self] sampleBuffer in
                guard let ciImage =  Processor.shared.process(sampleBuffer: sampleBuffer)
                else {
                    return
                }

                let uiImage = UIImage(ciImage: ciImage, scale: 1, orientation: .upMirrored)

                DispatchQueue.main.async { [weak self] in
                    self?.transferredView.image = uiImage
                }
            }
            .store(in: &cancellableBag)
    }

    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        let frame = Self.previewFrame(bounds: view.bounds, targetRatio: CGSize(width: 480, height: 640))
        previewView.frame = frame
        transferredView.frame = frame
    }

    override func viewLayoutMarginsDidChange() {
        super.viewLayoutMarginsDidChange()
        let frame = Self.previewFrame(bounds: view.bounds, targetRatio: CGSize(width: 480, height: 640))
        previewView.frame = frame
        transferredView.frame = frame
    }

    private static func previewFrame(bounds: CGRect, targetRatio: CGSize) -> CGRect {
        let woh = targetRatio.width / targetRatio.height
        if bounds.width / bounds.height < woh {
            let widthDiff = bounds.width - bounds.height * woh
            return bounds.insetBy(dx: widthDiff / 2, dy: 0)
        } else {
            let heightDiff = bounds.height - bounds.width / woh
            return bounds.insetBy(dx: 0, dy: heightDiff / 2)
        }
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)

        switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                DispatchQueue.global().async { [weak self] in
                    self?.session.startRunning()
                }
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                    if granted {
                        self?.session.startRunning()
                    }
                }
            default:
                break
        }
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)

        DispatchQueue.global().async { [weak self] in
            self?.session.stopRunning()
        }
    }

    private func setupSession() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        guard let videoDevice = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .unspecified)
        else { return }

        guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice),
              session.canAddInput(videoDeviceInput)
        else { return }

        session.addInput(videoDeviceInput)

        let photoOutput = AVCapturePhotoOutput()
        guard session.canAddOutput(photoOutput)
        else { return }

        session.sessionPreset = .vga640x480
        session.addOutput(photoOutput)

        let videoOutput = AVCaptureVideoDataOutput()
        guard session.canAddOutput(videoOutput)
        else { return }
        videoOutput.setSampleBufferDelegate(videoStream, queue: videoStream.queue)
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        session.addOutput(videoOutput)

        videoOutput.connection(with: .video)?.videoOrientation = .portraitUpsideDown
    }
}
