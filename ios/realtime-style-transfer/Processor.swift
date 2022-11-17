import AVFoundation
import CoreImage
import CoreML
import MetalPerformanceShaders

class Processor {
    static let shared = Processor()

    private let encoder = try! adain_vgg(configuration: MLModelConfiguration().then {
        $0.computeUnits = .all
        $0.allowLowPrecisionAccumulationOnGPU = true
    })

    private let decoder = try! adain_dec(configuration: MLModelConfiguration().then {
        $0.computeUnits = .all
        $0.allowLowPrecisionAccumulationOnGPU = true
    })

    private var modelInputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var modelOutputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var outputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .bgra8Unorm, usage: [.shaderRead, .shaderWrite])!

    var textureCache: CVMetalTextureCache!

    init() {
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, GPU.device, nil, &textureCache)
    }

    func process(sampleBuffer: CMSampleBuffer) -> CIImage? {
        guard let mtlTexture = getTexture(from: sampleBuffer),
              (try? mtlTexture.convert(into: modelInputBuffer.texture)) != nil,
              let input = modelInputBuffer.mlmultiarray,
              let latent = encode(input),
              let output = decode(latent)
        else {
            return nil
        }

        modelOutputBuffer.mlmultiarray = output

        guard (try? modelOutputBuffer.texture.convert(into: outputBuffer.texture)) != nil
        else {
            return nil
        }

        return CIImage(mtlTexture: outputBuffer.texture)
    }

    func getTexture(from sampleBuffer: CMSampleBuffer) -> MTLTexture? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else {
            return nil
        }

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
            return nil
        }

        let mtlTexture = CVMetalTextureGetTexture(texture)!

        return mtlTexture
    }

    func encode(_ input: MLMultiArray) -> MLMultiArray? {
        try? encoder.prediction(input: adain_vggInput(x: input)).var_171
    }

    func decode(_ input: MLMultiArray) -> MLMultiArray? {
        try? decoder.prediction(input: adain_decInput(x: input)).var_138
    }
}
