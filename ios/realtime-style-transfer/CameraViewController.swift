import AVFoundation
import Combine
import CoreML
import PhotosUI
import Then
import UIKit

class CameraViewController: UIViewController {
    private let session = AVCaptureSession()
    private let videoStream = VideoStream()
    private let previewView = PreviewView()
    private let transferredView = UIImageView()

    private var cancellableBag = Set<AnyCancellable>()

    private lazy var photoPickerButton = UIButton.systemButton(with: UIImage(systemName: "photo.on.rectangle")!, target: self, action: #selector(onPhotoPickerButton)).then {
        $0.tintColor = .white
        $0.layer.cornerRadius = 8
        $0.layer.cornerCurve = .continuous
        $0.layer.masksToBounds = true
    }

    private lazy var resetButton = UIButton.systemButton(
        with: UIImage(systemName: "xmark.bin")!,
        target: self,
        action: #selector(onResetButton)
    ).then {
        $0.tintColor = .white
    }

    private lazy var shareButton = UIButton.systemButton(
        with: UIImage(systemName: "square.and.arrow.up")!,
        target: self,
        action: #selector(onShareButton)
    ).then {
        $0.tintColor = .white
    }

    private lazy var stack = UIStackView(arrangedSubviews: [
        photoPickerButton,
        resetButton,
//        shareButton
    ]).then {
        $0.alignment = .center
        $0.distribution = .fillEqually
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        view.backgroundColor = .black

        view.addSubview(previewView)
        view.addSubview(transferredView)
        view.addSubview(stack)

        setupSession()

        videoStream.publisher
            .throttle(for: 0.3, scheduler: RunLoop.current, latest: true)
            .sink { [unowned self] (sampleBuffer) -> Void in
//                let start = CFAbsoluteTimeGetCurrent()
                guard let ciImage =  Processor.shared.process(sampleBuffer: sampleBuffer)
                else {
                    return
                }
//                let duration = CFAbsoluteTimeGetCurrent() - start
//                print(duration)

                let uiImage = UIImage(ciImage: ciImage)

                DispatchQueue.main.async { [weak self] in
                    self?.transferredView.image = uiImage
                }
            }
            .store(in: &cancellableBag)
    }

    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        let frame = previewFrame()
        previewView.frame = frame
        transferredView.frame = frame

        stack.frame = stackFrame()
    }

    override func viewLayoutMarginsDidChange() {
        super.viewLayoutMarginsDidChange()
        let frame = previewFrame()
        previewView.frame = frame
        transferredView.frame = frame

        stack.frame = stackFrame()
    }

    private func previewFrame() -> CGRect {
        let targetRatio = CGSize(width: 480, height: 640)
        let bounds = view.bounds

        let woh = targetRatio.width / targetRatio.height
        if bounds.width / bounds.height < woh {
            let widthDiff = bounds.width - bounds.height * woh
            return bounds.insetBy(dx: widthDiff / 2, dy: 0)
        } else {
            let heightDiff = bounds.height - bounds.width / woh
            return bounds.insetBy(dx: 0, dy: heightDiff / 2)
        }
    }

    private func stackFrame() -> CGRect {
        let safeArea = view.bounds.inset(by: view.safeAreaInsets)
        return safeArea.inset(by: UIEdgeInsets(top: safeArea.height - 80, left: 0, bottom: 0, right: 0))
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

        guard let videoDevice: AVCaptureDevice =
            AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .unspecified) ??
            AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .unspecified)
        else {
            return
        }

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

    @objc
    func onShareButton() {
        guard let image = transferredView.image
        else {
            return
        }

        transferredView.alpha = 0
        UIView.animate(withDuration: 0.3) {
            self.transferredView.alpha = 1
        }

        self.present(
            UIActivityViewController(activityItems: [image], applicationActivities: nil),
            animated: true
        )
    }

    @objc
    func onResetButton() {
        Processor.shared.discardStyle()
    }
}

extension CameraViewController: PHPickerViewControllerDelegate {
    @objc
    func onPhotoPickerButton() {
        var config = PHPickerConfiguration(photoLibrary: .shared())
        config.filter = PHPickerFilter.any(of: [.images, .livePhotos])
        config.preferredAssetRepresentationMode = .current
        config.selection = .default
        config.selectionLimit = 1

        let picker = PHPickerViewController(configuration: config)
        picker.delegate = self
        present(picker, animated: true)
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        dismiss(animated: true)

        guard let itemProvider = results.first?.itemProvider
        else {
            return
        }

        if itemProvider.canLoadObject(ofClass: UIImage.self) {
            itemProvider.loadObject(ofClass: UIImage.self) { photo, error in
                guard let image = photo as? UIImage
                else {
                    return
                }

                do {
                    try Processor.shared.encode(style: image)
                } catch {

                }
            }
        }
    }
}
