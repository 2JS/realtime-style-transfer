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

    private let encoder = try! adain_vgg(configuration: MLModelConfiguration().then {
        $0.computeUnits = .all
        $0.allowLowPrecisionAccumulationOnGPU = true
    })

    private let decoder = try! adain_dec(configuration: MLModelConfiguration().then {
        $0.computeUnits = .all
        $0.allowLowPrecisionAccumulationOnGPU = true
    })

    private var cancellableBag = Set<AnyCancellable>()

    private var modelInputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var modelOutputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var a = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .bgra8Unorm, usage: [.shaderRead, .shaderWrite])

    var textureCache: CVMetalTextureCache!

    override func viewDidLoad() {
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, GPU.device, nil, &textureCache)

        super.viewDidLoad()

        view.backgroundColor = .black

        view.addSubview(previewView)

        view.addSubview(transferredView)

        setupSession()

//        previewView.session = session

        videoStream.publisher
            .throttle(for: 0.066, scheduler: RunLoop.current, latest: true)
            .sink { [unowned self] sampleBuffer in
                let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)!

                let colorFormat: MTLPixelFormat = .bgra8Unorm
                let width = CVPixelBufferGetWidth(imageBuffer)
                let height = CVPixelBufferGetHeight(imageBuffer)

                var texture: CVMetalTexture!
                let status = CVMetalTextureCacheCreateTextureFromImage(
                    kCFAllocatorDefault,
                    self.textureCache,
                    imageBuffer,
                    nil,
                    colorFormat,
                    width,
                    height,
                    0,
                    &texture
                )


                if status != kCVReturnSuccess {

                }

                let mtlTexture = CVMetalTextureGetTexture(texture)!

                try! mtlTexture.convert(into: self.modelInputBuffer.texture)

                guard let input = self.modelInputBuffer.mlmultiarray
                else {
                    return
                }
                let start = CFAbsoluteTimeGetCurrent()
                let output = try! encoder.prediction(input: adain_vggInput(x: input)).var_171
                let duration = CFAbsoluteTimeGetCurrent() - start

                print(output, duration)

                let decoded = try! decoder.prediction(input: adain_decInput(x: output)).var_138

                print(decoded)

                modelOutputBuffer.mlmultiarray = decoded

                let uiImage = UIImage(ciImage: CIImage(mtlTexture: modelOutputBuffer.texture.converted(pixelFormat: .bgra8Unorm)!)!.oriented(.upMirrored))

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
