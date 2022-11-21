import AVFoundation
import CoreML
import MetalKit
import MetalPerformanceShaders
import UIKit


let width = 640
let height = 480

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

    private let loader = MTKTextureLoader(device: GPU.device)

    private var modelInputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var modelOutputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var outputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .bgra8Unorm, usage: [.shaderRead, .shaderWrite])!

    private var a = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .bgra8Unorm, usage: [.shaderRead, .shaderWrite])!
    var styleInputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var styleArray: MLMultiArray?

    var textureCache: CVMetalTextureCache!

    init() {
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, GPU.device, nil, &textureCache)
    }

    deinit {
        CVMetalTextureCacheFlush(textureCache, 0)
    }

    func process(sampleBuffer: CMSampleBuffer) -> CIImage? {
        guard let mtlTexture = getTexture(from: sampleBuffer),
              (try? mtlTexture.convert(into: modelInputBuffer.texture)) != nil,
              let input = modelInputBuffer.mlmultiarray,
              let latent = encode(input),
              let output = decode(content: latent, style: styleArray)
        else {
            return nil
        }

        modelOutputBuffer.mlmultiarray = output

        guard (try? modelOutputBuffer.texture.convert(into: outputBuffer.texture)) != nil
        else {
            return nil
        }

        return CIImage(mtlTexture: outputBuffer.texture)?.oriented(.upMirrored)
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

    func encode(style: UIImage) {
//        let ciImage: CIImage
//        if let image = style.ciImage {
//            ciImage = image
//        } else if let image = style.cgImage {
//            ciImage = CIImage(cgImage: image)
//        } else {
//            return
//        }
        guard let cgImage = style.cgImage
        else {
            return
        }

        guard let texture = try? loader.newTexture(cgImage: cgImage)
        else {
            return
        }

        guard let commandBuffer = GPU.queue.makeCommandBuffer()
        else {
            return
        }

        guard (try? texture.convert(into: styleInputBuffer.texture)) != nil
        else {
            return
        }

        guard let mlmultiarray = styleInputBuffer.mlmultiarray,
              let style = encode(mlmultiarray)
        else {
            return
        }

        self.styleArray = style
    }

    func encode(_ input: MLMultiArray) -> MLMultiArray? {
        try? encoder.prediction(x: input).var_171
    }

    func decode(content: MLMultiArray, style: MLMultiArray? = nil) -> MLMultiArray? {
        try? decoder.prediction(content: content, style: style ?? content).var_291
    }
}
